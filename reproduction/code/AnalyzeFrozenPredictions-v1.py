#!/usr/bin/env python
"""Subgroup, calibration, and error-candidate analysis for frozen predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--predictions-file", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--max-visible-bytes", type=int, default=512)
    parser.add_argument("--error-limit", type=int, default=50)
    return parser.parse_args()


def metrics(labels, scores, predictions):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    mixed = np.unique(labels).size == 2
    return {
        "N": int(labels.size),
        "PositiveN": int(labels.sum()),
        "NegativeN": int((labels == 0).sum()),
        "F1": f1_score(labels, predictions, zero_division=0),
        "P": precision_score(labels, predictions, zero_division=0),
        "R": recall_score(labels, predictions, zero_division=0),
        "Acc": accuracy_score(labels, predictions),
        "FPR": fp / max(1, fp + tn),
        "FNR": fn / max(1, fn + tp),
        "MCC": matthews_corrcoef(labels, predictions) if mixed else math.nan,
        "PRAUC": average_precision_score(labels, scores) if mixed else math.nan,
        "ROCAUC": roc_auc_score(labels, scores) if mixed else math.nan,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def calibration(labels, scores, bins=15):
    labels = np.asarray(labels, dtype=int)
    scores = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(scores, edges[1:-1]), bins - 1)
    ece = 0.0
    rows = []
    for index in range(bins):
        mask = indices == index
        if not np.any(mask):
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += float(mask.mean()) * abs(confidence - accuracy)
        rows.append(
            {
                "Bin": index,
                "N": int(mask.sum()),
                "MeanScore": confidence,
                "PositiveRate": accuracy,
                "AbsoluteGap": abs(confidence - accuracy),
            }
        )
    mixed = np.unique(labels).size == 2
    summary = {
        "ECE15": ece,
        "Brier": float(np.mean((scores - labels) ** 2)),
        "NLL": log_loss(labels, scores, labels=[0, 1]) if mixed else math.nan,
    }
    return summary, rows


def add_derived_groups(frame, visible_bytes):
    output = frame.copy()
    byte_lengths = output["text"].fillna("").astype(str).map(lambda value: len(value.encode("utf-8")))
    output["text_byte_length"] = byte_lengths
    output["length_bucket"] = pd.cut(
        byte_lengths,
        bins=[-1, 128, 512, 1024, np.inf],
        labels=["000-128", "129-512", "513-1024", "1025+"],
    ).astype(str)
    if {"span_start", "span_end"}.issubset(output.columns):
        starts = pd.to_numeric(output["span_start"], errors="coerce").fillna(-1).astype(int)
        ends = pd.to_numeric(output["span_end"], errors="coerce").fillna(-1).astype(int)
        visibility = np.where(
            starts < 0,
            "no_span",
            np.where(starts >= visible_bytes, "beyond_512", np.where(ends > visible_bytes, "partial_512", "full_512")),
        )
        output["span_visibility"] = visibility
        denominator = byte_lengths.clip(lower=1)
        ratio = starts.clip(lower=0) / denominator
        output["span_position"] = np.where(
            starts < 0,
            "no_span",
            np.where(ratio < 1 / 3, "early", np.where(ratio < 2 / 3, "middle", "late")),
        )
    return output


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.data_file).copy()
    predictions = pd.read_parquet(args.predictions_file).copy()
    if not {"id", "label", "Model", "Seed", "score", "prediction"}.issubset(predictions.columns):
        raise ValueError("Prediction file lacks required columns")
    source["id"] = source["id"].astype(str)
    predictions["id"] = predictions["id"].astype(str)
    source = add_derived_groups(source, args.max_visible_bytes)
    metadata_columns = [column for column in source.columns if column not in {"label"}]
    merged = predictions.drop(columns=[column for column in predictions if column in metadata_columns and column != "id"], errors="ignore").merge(
        source[metadata_columns], on="id", how="left", validate="many_to_one"
    )
    if merged["text"].isna().any():
        raise ValueError("Some prediction ids are absent from the source data")

    candidate_groups = [
        "duration_bucket",
        "sample_type",
        "augmentation",
        "category",
        "source",
        "severity",
        "length_bucket",
        "span_visibility",
        "span_position",
    ]
    group_columns = [column for column in candidate_groups if column in merged]
    subgroup_rows = []
    calibration_rows = []
    calibration_bins = []
    error_rows = []
    for (model, seed), part in merged.groupby(["Model", "Seed"], sort=True):
        values, bins = calibration(part["label"], part["score"])
        calibration_rows.append({"Model": model, "Seed": int(seed), **values})
        calibration_bins.extend({"Model": model, "Seed": int(seed), **row} for row in bins)
        for column in group_columns:
            for group, group_part in part.groupby(column, dropna=False):
                subgroup_rows.append(
                    {
                        "Model": model,
                        "Seed": int(seed),
                        "GroupColumn": column,
                        "Group": group,
                        **metrics(group_part["label"], group_part["score"], group_part["prediction"]),
                    }
                )
        false_positive = part[(part["label"] == 0) & (part["prediction"] == 1)].sort_values("score", ascending=False).head(args.error_limit)
        false_negative = part[(part["label"] == 1) & (part["prediction"] == 0)].sort_values("score", ascending=True).head(args.error_limit)
        for error_type, error_part in [("FP", false_positive), ("FN", false_negative)]:
            for rank, row in enumerate(error_part.itertuples(index=False), start=1):
                record = {
                    "Model": model,
                    "Seed": int(seed),
                    "ErrorType": error_type,
                    "Rank": rank,
                    "id": row.id,
                    "label": int(row.label),
                    "score": float(row.score),
                    "threshold": float(row.threshold),
                    "text": row.text,
                }
                for column in group_columns:
                    record[column] = getattr(row, column)
                error_rows.append(record)

    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(args.save_dir / "subgroup_metrics_by_seed.csv", index=False)
    subgroup_summary = (
        subgroup.groupby(["Model", "GroupColumn", "Group"], dropna=False, as_index=False)
        .agg(
            Seeds=("Seed", "nunique"),
            N=("N", "first"),
            F1Mean=("F1", "mean"),
            F1Std=("F1", "std"),
            RecallMean=("R", "mean"),
            FPRMean=("FPR", "mean"),
            FNRMean=("FNR", "mean"),
            MCCMean=("MCC", "mean"),
            PRAUCMean=("PRAUC", "mean"),
        )
    )
    subgroup_summary["F1Std"] = subgroup_summary["F1Std"].fillna(0.0)
    subgroup_summary.to_csv(args.save_dir / "subgroup_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(args.save_dir / "calibration_metrics.csv", index=False)
    pd.DataFrame(calibration_bins).to_csv(args.save_dir / "calibration_bins.csv", index=False)
    pd.DataFrame(error_rows).to_csv(args.save_dir / "error_candidates.csv", index=False)
    (args.save_dir / "analysis_protocol.json").write_text(
        json.dumps(
            {
                "data_file": str(args.data_file),
                "predictions_file": str(args.predictions_file),
                "max_visible_bytes": args.max_visible_bytes,
                "group_columns": group_columns,
                "error_candidates_note": "Candidates for manual coding; they are not automatically assigned semantic error categories.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote subgroup and calibration analysis to {args.save_dir}")


if __name__ == "__main__":
    main()

