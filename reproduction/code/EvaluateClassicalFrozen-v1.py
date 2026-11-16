#!/usr/bin/env python
"""Evaluate already-fitted classical baselines with frozen validation thresholds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-csv", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--operating-point", choices=["Safety10_15", "F1Optimal"], default="Safety10_15")
    parser.add_argument("--group-columns", default="duration_bucket,sample_type,augmentation,category,source,severity")
    return parser.parse_args()


def metric_row(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = scores >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    mixed = np.unique(labels).size == 2
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return {
        "N": int(labels.size),
        "Threshold": float(threshold),
        "F1": f1_score(labels, predictions, zero_division=0),
        "P": precision_score(labels, predictions, zero_division=0),
        "R": recall_score(labels, predictions, zero_division=0),
        "Acc": accuracy_score(labels, predictions),
        "FPR": fp / max(1, fp + tn),
        "FNR": fn / max(1, fn + tp),
        "MCC": matthews_corrcoef(labels, predictions) if mixed else math.nan,
        "PRAUC": average_precision_score(labels, scores) if mixed else math.nan,
        "ROCAUC": roc_auc_score(labels, scores) if mixed else math.nan,
        "Brier": float(np.mean((scores - labels) ** 2)),
        "NLL": log_loss(labels, clipped, labels=[0, 1]) if mixed else math.nan,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def classifier_scores(classifier, features):
    # The stored models are binary linear classifiers. Computing their margin
    # directly also makes the artifact robust to harmless scikit-learn API
    # changes between the training and evaluation environments.
    coefficients = np.asarray(classifier.coef_, dtype=float)
    intercept = np.asarray(classifier.intercept_, dtype=float)
    margins = np.asarray(features @ coefficients.T, dtype=float).reshape(-1) + float(intercept[0])
    return expit(margins)


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.data_file).reset_index(drop=True).copy()
    if not {"id", "text", "label"}.issubset(frame.columns):
        raise ValueError("The evaluation file must contain id, text, and label columns")
    frame["id"] = frame["id"].astype(str)
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(int)
    labels = frame["label"].to_numpy(dtype=int)
    groups = [value.strip() for value in args.group_columns.split(",") if value.strip() in frame]

    runs = pd.read_csv(args.run_csv).drop_duplicates(["Model", "Seed"], keep="last")
    threshold_column = "SafetyThreshold" if args.operating_point == "Safety10_15" else "F1Threshold"
    if threshold_column not in runs:
        raise ValueError(f"Missing threshold column {threshold_column} in {args.run_csv}")

    prediction_rows = []
    metric_rows = []
    subgroup_rows = []
    for run in runs.itertuples(index=False):
        model_name = str(run.Model)
        seed = int(run.Seed)
        artifact = args.model_dir / f"{model_name}_seed{seed}.joblib"
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        saved = joblib.load(artifact)
        features = saved["vectorizer"].transform(frame["text"])
        scores = classifier_scores(saved["classifier"], features)
        threshold = float(getattr(run, threshold_column))
        values = metric_row(labels, scores, threshold)
        metric_rows.append(
            {
                "Model": model_name,
                "Seed": seed,
                "Source": f"classical_{args.operating_point}",
                "ArtifactMB": artifact.stat().st_size / 1024**2,
                "TrainableParams": int(getattr(run, "TrainableParams", 0)),
                **values,
            }
        )
        for index, score in enumerate(scores):
            row = {
                "id": frame.iloc[index]["id"],
                "label": int(labels[index]),
                "Model": model_name,
                "Seed": seed,
                "score": float(score),
                "threshold": threshold,
                "prediction": int(score >= threshold),
            }
            for column in groups:
                row[column] = frame.iloc[index][column]
            prediction_rows.append(row)
        for column in groups:
            for group, indexes in frame.groupby(column, dropna=False).groups.items():
                index_array = np.asarray(list(indexes), dtype=int)
                subgroup_rows.append(
                    {
                        "GroupColumn": column,
                        "Group": group,
                        "Model": model_name,
                        "Seed": seed,
                        **metric_row(labels[index_array], scores[index_array], threshold),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(args.save_dir / "classical_external_metrics.csv", index=False)
    predictions.to_parquet(args.save_dir / "classical_external_predictions.parquet", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.save_dir / "classical_external_subgroups.csv", index=False)
    summary = (
        metrics.groupby(["Model", "Source"], as_index=False)
        .agg(
            Seeds=("Seed", "nunique"),
            F1Mean=("F1", "mean"),
            F1Std=("F1", "std"),
            PrecisionMean=("P", "mean"),
            RecallMean=("R", "mean"),
            FPRMean=("FPR", "mean"),
            MCCMean=("MCC", "mean"),
            PRAUCMean=("PRAUC", "mean"),
            ArtifactMB=("ArtifactMB", "mean"),
            TrainableParams=("TrainableParams", "mean"),
        )
        .sort_values("F1Mean", ascending=False)
    )
    summary["F1Std"] = summary["F1Std"].fillna(0.0)
    summary.to_csv(args.save_dir / "classical_external_summary.csv", index=False)
    (args.save_dir / "protocol.json").write_text(
        json.dumps(
            {
                "data_file": str(args.data_file),
                "rows": len(frame),
                "operating_point": args.operating_point,
                "threshold_source": "original validation data; never refit on this evaluation file",
                "models": int(len(metrics)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
