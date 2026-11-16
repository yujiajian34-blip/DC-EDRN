#!/usr/bin/env python
"""Bootstrap confidence intervals and paired McNemar tests from frozen predictions."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def percentile_ci(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def stratified_bootstrap(labels, predictions, samples, rng):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    values = np.empty(samples, dtype=float)
    for iteration in range(samples):
        selected = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        values[iteration] = f1_score(labels[selected], predictions[selected], zero_division=0)
    return values


def holm_adjust(pvalues):
    if not pvalues:
        return []
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * float(pvalues[index])))
        adjusted[index] = running
    return adjusted.tolist()


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.predictions)
    required = {"id", "label", "Model", "Seed", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    rng = np.random.default_rng(args.seed)
    ci_rows = []
    for (model, seed), group in frame.groupby(["Model", "Seed"]):
        values = stratified_bootstrap(group.label, group.prediction, args.bootstrap_samples, rng)
        low, high = percentile_ci(values)
        ci_rows.append({
            "Model": model, "Seed": int(seed), "N": len(group),
            "F1": f1_score(group.label, group.prediction, zero_division=0),
            "F1CI95Low": low, "F1CI95High": high,
        })

    tests = []
    reference = frame[frame.Model.eq(args.reference_model)]
    if reference.empty:
        raise ValueError(f"Reference model not found: {args.reference_model}")
    for ref_seed, ref_group in reference.groupby("Seed"):
        ref = ref_group[["id", "label", "prediction"]].rename(columns={"prediction": "ref_prediction"})
        for (model, seed), candidate in frame[~frame.Model.eq(args.reference_model)].groupby(["Model", "Seed"]):
            if int(seed) not in {0, int(ref_seed)}:
                continue
            merged = ref.merge(candidate[["id", "label", "prediction"]], on=["id", "label"], how="inner", validate="one_to_one")
            if len(merged) != len(ref):
                raise ValueError(f"Prediction IDs do not align: {args.reference_model} seed {ref_seed} vs {model} seed {seed}")
            ref_correct = merged.ref_prediction.to_numpy() == merged.label.to_numpy()
            cand_correct = merged.prediction.to_numpy() == merged.label.to_numpy()
            ref_only = int(np.sum(ref_correct & ~cand_correct))
            candidate_only = int(np.sum(~ref_correct & cand_correct))
            discordant = ref_only + candidate_only
            pvalue = float(binomtest(min(ref_only, candidate_only), discordant, 0.5).pvalue) if discordant else 1.0
            tests.append({
                "ReferenceModel": args.reference_model, "ReferenceSeed": int(ref_seed),
                "CandidateModel": model, "CandidateSeed": int(seed), "N": len(merged),
                "ReferenceOnlyCorrect": ref_only, "CandidateOnlyCorrect": candidate_only,
                "Discordant": discordant, "McNemarExactP": pvalue,
            })
    adjusted = holm_adjust([row["McNemarExactP"] for row in tests])
    for row, value in zip(tests, adjusted):
        row["McNemarHolmP"] = value
    pd.DataFrame(ci_rows).to_csv(args.save_dir / "bootstrap_f1_ci.csv", index=False)
    pd.DataFrame(tests).to_csv(args.save_dir / "mcnemar_tests.csv", index=False)
    metadata = {"predictions": str(args.predictions), "bootstrap_samples": args.bootstrap_samples, "random_seed": args.seed, "reference_model": args.reference_model}
    (args.save_dir / "statistical_protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(ci_rows).to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
