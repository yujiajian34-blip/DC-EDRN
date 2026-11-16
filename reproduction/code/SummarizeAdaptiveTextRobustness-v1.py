#!/usr/bin/env python
"""Summarize clean-to-perturbed degradation from cached frozen predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-root", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    return p.parse_args()


def metrics(part):
    labels = part.label.to_numpy(dtype=int)
    pred = part.prediction.to_numpy(dtype=int)
    neg, pos = labels == 0, labels == 1
    return pd.Series({
        "N": len(part), "F1": f1_score(labels, pred, zero_division=0),
        "P": precision_score(labels, pred, zero_division=0),
        "R": recall_score(labels, pred, zero_division=0),
        "Acc": accuracy_score(labels, pred),
        "FPR": float(pred[neg].mean()) if neg.any() else np.nan,
        "FNR": float((1 - pred[pos]).mean()) if pos.any() else np.nan,
        "PRAUC": average_precision_score(labels, part.score) if len(np.unique(labels)) == 2 else np.nan,
    })


def main():
    args = parse_args(); args.save_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in sorted(args.evaluation_root.glob("*/external_predictions.parquet")):
        frame = pd.read_parquet(path)
        if "attack_family" not in frame:
            frame["attack_family"] = path.parent.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No completed prediction files under {args.evaluation_root}")
    data = pd.concat(frames, ignore_index=True)
    if "preprocessing" not in data:
        data["preprocessing"] = "raw"
    grouped = data.groupby(["Model", "Seed", "preprocessing", "attack_family", "severity"], dropna=False).apply(metrics, include_groups=False).reset_index()
    clean = grouped[grouped.attack_family.eq("clean")][["Model", "Seed", "preprocessing", "F1", "R", "FPR"]].rename(columns={"F1":"CleanF1", "R":"CleanR", "FPR":"CleanFPR"})
    grouped = grouped.merge(clean, on=["Model", "Seed", "preprocessing"], how="left")
    grouped["F1Drop"] = grouped.CleanF1 - grouped.F1
    grouped["RecallDrop"] = grouped.CleanR - grouped.R
    grouped["FPRIncrease"] = grouped.FPR - grouped.CleanFPR
    grouped.to_csv(args.save_dir / "robustness_by_model_seed.csv", index=False)

    mean = grouped.groupby(["Model", "preprocessing", "attack_family", "severity"], as_index=False).agg(
        Seeds=("Seed", "nunique"), F1Mean=("F1", "mean"), F1SD=("F1", "std"),
        RecallMean=("R", "mean"), FPRMean=("FPR", "mean"), PRAUCMean=("PRAUC", "mean"),
        F1DropMean=("F1Drop", "mean"), RecallDropMean=("RecallDrop", "mean"),
        FPRIncreaseMean=("FPRIncrease", "mean"),
    )
    mean.to_csv(args.save_dir / "robustness_by_attack.csv", index=False)
    high = mean[mean.severity.eq("high")].copy()
    high.to_csv(args.save_dir / "paper_table_high_severity.csv", index=False)

    summary_rows = []
    for (model, preprocessing), part in mean.groupby(["Model", "preprocessing"]):
        clean_row = part[part.attack_family.eq("clean")].iloc[0]
        attacks = part[(part.severity.eq("high")) & (~part.attack_family.eq("clean"))]
        summary_rows.append({
            "Model": model, "Preprocessing": preprocessing, "CleanF1": clean_row.F1Mean,
            "HighSeverityMeanF1": attacks.F1Mean.mean(),
            "HighSeverityWorstF1": attacks.F1Mean.min(),
            "HighSeverityMeanRecall": attacks.RecallMean.mean(),
            "HighSeverityWorstRecall": attacks.RecallMean.min(),
            "HighSeverityMaxFPR": attacks.FPRMean.max(),
            "MeanF1Drop": attacks.F1DropMean.mean(),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["HighSeverityWorstRecall", "HighSeverityMaxFPR"], ascending=[False, True])
    summary.to_csv(args.save_dir / "robustness_model_summary.csv", index=False)
    (args.save_dir / "summary_protocol.json").write_text(json.dumps({
        "prediction_files": len(frames), "prediction_rows": len(data),
        "reporting_rule": "Primary robustness quantities are clean-to-perturbed recall loss and benign FPR increase; high-severity worst-family results are reported without retuning thresholds.",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
