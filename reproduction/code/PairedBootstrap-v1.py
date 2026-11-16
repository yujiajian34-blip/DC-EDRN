#!/usr/bin/env python
"""Paired bootstrap confidence intervals and McNemar tests on frozen predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-files", required=True, help="Comma-separated parquet files")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def collapse_seeds(frame):
    grouped = frame.groupby(["Model", "id"], as_index=False).agg(
        label=("label", "first"),
        score=("score", "mean"),
        vote=("prediction", "mean"),
        seeds=("Seed", "nunique"),
    )
    grouped["prediction"] = (grouped["vote"] >= 0.5).astype(int)
    return grouped


def ci(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_f1(labels, predictions, indices):
    sampled_labels = labels[indices]
    sampled_predictions = predictions[indices]
    tp = np.sum((sampled_labels == 1) & (sampled_predictions == 1), axis=1)
    fp = np.sum((sampled_labels == 0) & (sampled_predictions == 1), axis=1)
    fn = np.sum((sampled_labels == 1) & (sampled_predictions == 0), axis=1)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)


def family_budget(name):
    match = re.match(r"Budget(55K|115K|350K)_(.+)", str(name))
    return match.group(1) if match else None


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(value.strip()) for value in args.prediction_files.split(",") if value.strip()]
    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(["Model", "Seed", "id"], keep="last")
    collapsed = collapse_seeds(combined)
    rng = np.random.default_rng(args.seed)

    model_rows = []
    model_lookup = {}
    for model, part in collapsed.groupby("Model"):
        part = part.sort_values("id").reset_index(drop=True)
        labels = part["label"].to_numpy(dtype=int)
        predictions = part["prediction"].to_numpy(dtype=int)
        n = len(part)
        samples = rng.integers(0, n, size=(args.resamples, n))
        bootstrap = bootstrap_f1(labels, predictions, samples)
        low, high = ci(bootstrap)
        model_rows.append(
            {
                "Model": model,
                "N": n,
                "Seeds": int(part["seeds"].max()),
                "F1": f1_score(labels, predictions, zero_division=0),
                "F1CI95Low": low,
                "F1CI95High": high,
            }
        )
        model_lookup[model] = part[["id", "label", "prediction"]]

    references = [
        model for model in model_lookup
        if model.endswith("_DC_EDRN") or model.endswith("_Full_EDRN") or model == "Paper_NoExplicitDuration"
    ]
    comparison_rows = []
    for reference in references:
        budget = family_budget(reference)
        if budget is not None:
            candidates = [model for model in model_lookup if family_budget(model) == budget and model != reference]
        else:
            candidates = [model for model in model_lookup if model != reference and family_budget(model) is None]
        for candidate in candidates:
            merged = model_lookup[reference].merge(
                model_lookup[candidate], on=["id", "label"], suffixes=("_ref", "_candidate"), validate="one_to_one"
            )
            if merged.empty:
                continue
            labels = merged["label"].to_numpy(dtype=int)
            pred_ref = merged["prediction_ref"].to_numpy(dtype=int)
            pred_candidate = merged["prediction_candidate"].to_numpy(dtype=int)
            n = len(merged)
            indices = rng.integers(0, n, size=(args.resamples, n))
            differences = bootstrap_f1(labels, pred_ref, indices) - bootstrap_f1(labels, pred_candidate, indices)
            low, high = ci(differences)
            correct_ref = pred_ref == labels
            correct_candidate = pred_candidate == labels
            ref_only = int(np.sum(correct_ref & ~correct_candidate))
            candidate_only = int(np.sum(~correct_ref & correct_candidate))
            discordant = ref_only + candidate_only
            p_value = binomtest(min(ref_only, candidate_only), discordant, 0.5).pvalue if discordant else 1.0
            comparison_rows.append(
                {
                    "Reference": reference,
                    "Candidate": candidate,
                    "N": n,
                    "ReferenceF1": f1_score(labels, pred_ref, zero_division=0),
                    "CandidateF1": f1_score(labels, pred_candidate, zero_division=0),
                    "F1Difference": f1_score(labels, pred_ref, zero_division=0) - f1_score(labels, pred_candidate, zero_division=0),
                    "DifferenceCI95Low": low,
                    "DifferenceCI95High": high,
                    "ReferenceOnlyCorrect": ref_only,
                    "CandidateOnlyCorrect": candidate_only,
                    "McNemarExactP": p_value,
                }
            )

    pd.DataFrame(model_rows).sort_values("F1", ascending=False).to_csv(args.save_dir / "model_f1_bootstrap_ci.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(args.save_dir / "paired_f1_and_mcnemar.csv", index=False)
    (args.save_dir / "protocol.json").write_text(
        json.dumps(
            {
                "prediction_files": [str(path) for path in paths],
                "resamples": args.resamples,
                "seed": args.seed,
                "seed_aggregation": "majority vote per id before paired resampling",
                "sampling_unit": "example id",
                "multiple_comparison_note": "Report exact p-values and apply Holm correction in the paper table if many comparisons are retained.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote paired statistics to {args.save_dir}")


if __name__ == "__main__":
    main()
