#!/usr/bin/env python
"""Create paper-ready uncertainty tests and qualitative error samples."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-predictions", type=Path, required=True)
    parser.add_argument("--error-predictions", type=Path, required=True)
    parser.add_argument("--error-data", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--reference-model", default="Paper_NoExplicitDuration")
    parser.add_argument(
        "--baseline-model",
        default="auto",
        help="Model name, or auto for the highest mean-F1 non-Paper model.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--samples-per-error", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def validate_predictions(frame, name):
    required = {"id", "label", "Model", "Seed", "score", "threshold", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} predictions are missing: {sorted(missing)}")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["Seed"] = frame["Seed"].astype(int)
    frame["prediction"] = frame["prediction"].astype(int)
    duplicated = frame.duplicated(["Model", "Seed", "id"])
    if duplicated.any():
        raise ValueError(f"{name} predictions contain duplicate model/seed/id rows")
    return frame


def model_mean_f1(frame):
    rows = []
    for (model, seed), part in frame.groupby(["Model", "Seed"], sort=False):
        rows.append(
            {
                "Model": str(model),
                "Seed": int(seed),
                "F1": f1_score(part.label, part.prediction, zero_division=0),
            }
        )
    per_seed = pd.DataFrame(rows)
    return per_seed, per_seed.groupby("Model")["F1"].mean().sort_values(ascending=False)


def select_baseline(frame, reference, requested):
    _, means = model_mean_f1(frame)
    if reference not in means.index:
        raise ValueError(f"Reference model not found: {reference}")
    if requested != "auto":
        if requested not in means.index:
            raise ValueError(f"Baseline model not found: {requested}")
        return requested, means
    candidates = means[
        (means.index != reference)
        & ~means.index.to_series().str.startswith("Paper_", na=False).to_numpy()
    ]
    if candidates.empty:
        candidates = means[means.index != reference]
    if candidates.empty:
        raise ValueError("No baseline model is available")
    return str(candidates.index[0]), means


def stratified_indices(labels, rng):
    labels = np.asarray(labels, dtype=int)
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    return np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])


def quantiles(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def aligned_seed_frames(frame, reference, baseline):
    ref = frame[frame.Model.eq(reference)]
    base = frame[frame.Model.eq(baseline)]
    seeds = sorted(set(ref.Seed) & set(base.Seed))
    if not seeds:
        raise ValueError(f"No shared seeds between {reference} and {baseline}")
    aligned = {}
    for seed in seeds:
        left = ref[ref.Seed.eq(seed)][["id", "label", "prediction"]].rename(
            columns={"prediction": "reference_prediction"}
        )
        right = base[base.Seed.eq(seed)][["id", "label", "prediction"]].rename(
            columns={"prediction": "baseline_prediction"}
        )
        merged = left.merge(right, on=["id", "label"], how="inner", validate="one_to_one")
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError(f"Prediction IDs do not align for seed {seed}")
        aligned[seed] = merged.sort_values("id").reset_index(drop=True)
    return aligned


def confidence_intervals(frame, samples, rng):
    rows = []
    distributions = {}
    for (model, seed), part in frame.groupby(["Model", "Seed"], sort=False):
        part = part.sort_values("id").reset_index(drop=True)
        labels = part.label.to_numpy()
        predictions = part.prediction.to_numpy()
        f1_values = np.empty(samples)
        accuracy_values = np.empty(samples)
        for iteration in range(samples):
            selected = stratified_indices(labels, rng)
            f1_values[iteration] = f1_score(
                labels[selected], predictions[selected], zero_division=0
            )
            accuracy_values[iteration] = accuracy_score(labels[selected], predictions[selected])
        f1_low, f1_high = quantiles(f1_values)
        acc_low, acc_high = quantiles(accuracy_values)
        rows.append(
            {
                "Model": model,
                "Seed": int(seed),
                "N": len(part),
                "F1": f1_score(labels, predictions, zero_division=0),
                "F1_CI95_Low": f1_low,
                "F1_CI95_High": f1_high,
                "Accuracy": accuracy_score(labels, predictions),
                "Accuracy_CI95_Low": acc_low,
                "Accuracy_CI95_High": acc_high,
            }
        )
        distributions[(str(model), int(seed))] = (f1_values, accuracy_values)
    return pd.DataFrame(rows), distributions


def macro_seed_intervals(per_seed, distributions):
    rows = []
    for model, part in per_seed.groupby("Model", sort=False):
        seeds = sorted(part.Seed.astype(int).tolist())
        f1_matrix = np.vstack([distributions[(str(model), seed)][0] for seed in seeds])
        acc_matrix = np.vstack([distributions[(str(model), seed)][1] for seed in seeds])
        f1_boot = f1_matrix.mean(axis=0)
        acc_boot = acc_matrix.mean(axis=0)
        f1_low, f1_high = quantiles(f1_boot)
        acc_low, acc_high = quantiles(acc_boot)
        rows.append(
            {
                "Model": model,
                "Seeds": len(seeds),
                "F1_Mean": float(part.F1.mean()),
                "F1_Seed_SD": float(part.F1.std(ddof=1)) if len(part) > 1 else 0.0,
                "F1_Hierarchical_CI95_Low": f1_low,
                "F1_Hierarchical_CI95_High": f1_high,
                "Accuracy_Mean": float(part.Accuracy.mean()),
                "Accuracy_Seed_SD": float(part.Accuracy.std(ddof=1)) if len(part) > 1 else 0.0,
                "Accuracy_Hierarchical_CI95_Low": acc_low,
                "Accuracy_Hierarchical_CI95_High": acc_high,
            }
        )
    return pd.DataFrame(rows).sort_values("F1_Mean", ascending=False)


def paired_tests(aligned, reference, baseline, samples, rng):
    per_seed = []
    f1_differences = np.empty((len(aligned), samples))
    accuracy_differences = np.empty((len(aligned), samples))
    pvalues = []
    for row_index, (seed, part) in enumerate(aligned.items()):
        labels = part.label.to_numpy()
        ref_pred = part.reference_prediction.to_numpy()
        base_pred = part.baseline_prediction.to_numpy()
        for iteration in range(samples):
            selected = stratified_indices(labels, rng)
            f1_differences[row_index, iteration] = f1_score(
                labels[selected], ref_pred[selected], zero_division=0
            ) - f1_score(labels[selected], base_pred[selected], zero_division=0)
            accuracy_differences[row_index, iteration] = accuracy_score(
                labels[selected], ref_pred[selected]
            ) - accuracy_score(labels[selected], base_pred[selected])

        ref_correct = ref_pred == labels
        base_correct = base_pred == labels
        ref_only = int(np.sum(ref_correct & ~base_correct))
        base_only = int(np.sum(~ref_correct & base_correct))
        discordant = ref_only + base_only
        pvalue = (
            float(binomtest(min(ref_only, base_only), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        pvalues.append(pvalue)
        f1_low, f1_high = quantiles(f1_differences[row_index])
        acc_low, acc_high = quantiles(accuracy_differences[row_index])
        per_seed.append(
            {
                "ReferenceModel": reference,
                "BaselineModel": baseline,
                "Seed": int(seed),
                "N": len(part),
                "ReferenceF1": f1_score(labels, ref_pred, zero_division=0),
                "BaselineF1": f1_score(labels, base_pred, zero_division=0),
                "F1_Difference": f1_score(labels, ref_pred, zero_division=0)
                - f1_score(labels, base_pred, zero_division=0),
                "F1_Difference_CI95_Low": f1_low,
                "F1_Difference_CI95_High": f1_high,
                "Accuracy_Difference_CI95_Low": acc_low,
                "Accuracy_Difference_CI95_High": acc_high,
                "ReferenceOnlyCorrect": ref_only,
                "BaselineOnlyCorrect": base_only,
                "DiscordantPairs": discordant,
                "McNemar_Exact_P": pvalue,
            }
        )

    # Holm adjustment across the three seed-level McNemar tests.
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    for row, value in zip(per_seed, adjusted):
        row["McNemar_Holm_P"] = float(value)

    macro_f1 = f1_differences.mean(axis=0)
    macro_accuracy = accuracy_differences.mean(axis=0)
    f1_low, f1_high = quantiles(macro_f1)
    acc_low, acc_high = quantiles(macro_accuracy)
    macro = {
        "ReferenceModel": reference,
        "BaselineModel": baseline,
        "Seeds": len(aligned),
        "MeanF1Difference": float(
            np.mean([row["F1_Difference"] for row in per_seed])
        ),
        "MeanF1Difference_CI95_Low": f1_low,
        "MeanF1Difference_CI95_High": f1_high,
        "MeanAccuracyDifference_CI95_Low": acc_low,
        "MeanAccuracyDifference_CI95_High": acc_high,
        "F1DifferenceSignificant": bool(f1_low > 0 or f1_high < 0),
    }
    return pd.DataFrame(per_seed), pd.DataFrame([macro])


def suggested_error_type(row):
    text = str(row.get("text", ""))
    split = str(row.get("audit_split", ""))
    label = int(row["label"])
    span_start = pd.to_numeric(pd.Series([row.get("span_start", -1)]), errors="coerce").iloc[0]
    span_start = -1 if pd.isna(span_start) else int(span_start)
    byte_span_start = len(text[:span_start].encode("utf-8")) if span_start >= 0 else -1
    non_ascii_ratio = sum(ord(char) > 127 for char in text) / max(1, len(text))
    encoded = bool(re.search(r"(?:[A-Fa-f0-9]{32,}|[A-Za-z0-9+/]{48,}={0,2})", text))
    markup = bool(re.search(r"</?[A-Za-z][^>]*>|```|\{\s*[\"']|(?:def|class|SELECT)\s+\w+", text))
    quoted = bool(re.search(r"[\"'](?:ignore|system|assistant|instruction)", text, re.I))

    if label == 0:
        if split == "diagnostic_overdefense":
            return "FP_benign_instruction_overdefense"
        if split == "diagnostic_notinject":
            return "FP_benign_security_discussion"
        if markup:
            return "FP_code_or_markup"
        if quoted:
            return "FP_quoted_instruction"
        if len(text.encode("utf-8")) > 512:
            return "FP_long_benign_context"
        return "FP_instruction_like_benign_text"
    if byte_span_start >= 512:
        return "FN_attack_beyond_first_window"
    if split == "diagnostic_obfuscation" or encoded:
        return "FN_obfuscated_or_encoded_attack"
    if split == "diagnostic_long_context" or len(text.encode("utf-8")) > 512:
        return "FN_long_context"
    if non_ascii_ratio > 0.20:
        return "FN_multilingual_or_unicode"
    span_text = str(row.get("span_text", ""))
    if 0 < len(span_text.encode("utf-8")) <= 32:
        return "FN_short_or_implicit_attack"
    return "FN_indirect_or_unseen_attack"


def make_error_samples(predictions, data, reference, limit):
    ref = predictions[predictions.Model.eq(reference)].copy()
    if ref.empty:
        raise ValueError(f"Reference model absent from error predictions: {reference}")
    data = data.copy()
    data["id"] = data["id"].astype(str)
    metadata_columns = [
        column
        for column in [
            "id", "text", "audit_split", "sample_type", "category", "augmentation",
            "source", "duration_bucket", "span_text", "span_start", "span_end",
        ]
        if column in data.columns
    ]
    metadata = data[metadata_columns].drop_duplicates("id")
    ref["wrong"] = ref.prediction.ne(ref.label).astype(int)
    ref["signed_margin"] = ref.score.astype(float) - ref.threshold.astype(float)
    grouped = (
        ref.groupby(["id", "label"], as_index=False)
        .agg(
            evaluated_seeds=("Seed", "nunique"),
            wrong_seed_count=("wrong", "sum"),
            mean_score=("score", "mean"),
            mean_threshold=("threshold", "mean"),
            mean_signed_margin=("signed_margin", "mean"),
            wrong_seeds=("Seed", lambda values: ",".join(map(str, sorted(set(values))))),
        )
    )
    # Replace wrong_seeds with the actual subset of seeds that erred.
    wrong_seed_map = (
        ref[ref.wrong.eq(1)].groupby("id")["Seed"]
        .apply(lambda values: ",".join(map(str, sorted(set(values)))))
        .to_dict()
    )
    grouped["wrong_seeds"] = grouped.id.map(wrong_seed_map).fillna("")
    errors = grouped[grouped.wrong_seed_count.gt(0)].merge(
        metadata, on="id", how="left", validate="one_to_one"
    )
    errors["error_type"] = np.where(errors.label.eq(0), "FP", "FN")
    errors["error_strength"] = np.where(
        errors.label.eq(0), errors.mean_signed_margin, -errors.mean_signed_margin
    )
    errors = errors.sort_values(
        ["error_type", "wrong_seed_count", "error_strength"],
        ascending=[True, False, False],
    )
    selected = pd.concat(
        [errors[errors.error_type.eq(kind)].head(limit) for kind in ("FP", "FN")],
        ignore_index=True,
    )
    selected.insert(0, "case_number", range(1, len(selected) + 1))
    selected["robust_error"] = selected.wrong_seed_count.eq(selected.evaluated_seeds)
    selected["suggested_error_category"] = selected.apply(suggested_error_type, axis=1)
    selected["manual_error_category"] = ""
    selected["include_in_paper"] = ""
    selected["manual_notes"] = ""
    selected["text_excerpt"] = selected.get("text", pd.Series(dtype=str)).astype(str).str.slice(0, 1000)
    return selected, errors


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    stats = validate_predictions(pd.read_parquet(args.statistics_predictions), "statistics")
    errors = validate_predictions(pd.read_parquet(args.error_predictions), "error-audit")
    error_data = pd.read_parquet(args.error_data)

    baseline, model_means = select_baseline(
        stats, args.reference_model, args.baseline_model
    )
    rng = np.random.default_rng(args.seed)
    per_seed_ci, distributions = confidence_intervals(
        stats, args.bootstrap_samples, rng
    )
    macro_ci = macro_seed_intervals(per_seed_ci, distributions)
    aligned = aligned_seed_frames(stats, args.reference_model, baseline)
    paired_seed, paired_macro = paired_tests(
        aligned, args.reference_model, baseline, args.bootstrap_samples, rng
    )
    samples, all_errors = make_error_samples(
        errors, error_data, args.reference_model, args.samples_per_error
    )

    per_seed_ci.to_csv(args.save_dir / "model_seed_confidence_intervals.csv", index=False)
    macro_ci.to_csv(args.save_dir / "model_mean_confidence_intervals.csv", index=False)
    paired_seed.to_csv(args.save_dir / "paired_tests_by_seed.csv", index=False)
    paired_macro.to_csv(args.save_dir / "paired_bootstrap_summary.csv", index=False)
    samples.to_csv(
        args.save_dir / "manual_error_analysis_100.csv", index=False, encoding="utf-8-sig"
    )
    auto_summary = (
        samples.groupby(["error_type", "suggested_error_category"], as_index=False)
        .agg(Cases=("id", "size"), RobustErrors=("robust_error", "sum"))
        .sort_values(["error_type", "Cases"], ascending=[True, False])
    )
    auto_summary.to_csv(args.save_dir / "automatic_error_category_summary.csv", index=False)

    available = all_errors.groupby("error_type").size().to_dict()
    selected_counts = samples.groupby("error_type").size().to_dict()
    macro_row = paired_macro.iloc[0].to_dict()
    report = {
        "reference_model": args.reference_model,
        "automatically_selected_strongest_baseline": baseline,
        "model_mean_f1": {str(k): float(v) for k, v in model_means.items()},
        "paired_macro_result": macro_row,
        "bootstrap_samples": args.bootstrap_samples,
        "error_cases_available": {str(k): int(v) for k, v in available.items()},
        "error_cases_selected": {str(k): int(v) for k, v in selected_counts.items()},
        "interpretation": (
            "The reference-model F1 improvement is statistically supported when the "
            "paired 95% CI excludes zero. McNemar p-values are seed-level secondary tests."
        ),
        "manual_analysis_required": (
            "Review every selected row and fill manual_error_category, include_in_paper, "
            "and manual_notes. Suggested categories are heuristic and must not be reported "
            "as human annotation without review."
        ),
    }
    (args.save_dir / "paper_evidence_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    taxonomy = """# Error-analysis taxonomy

Review `manual_error_analysis_100.csv` and fill the three blank columns.

Suggested FP categories: benign instruction over-defense; benign security discussion;
code/markup; quoted instruction; long benign context; instruction-like benign text; other.

Suggested FN categories: attack beyond first 512-byte window; obfuscated/encoded attack;
long context; multilingual/Unicode; short/implicit attack; indirect/unseen attack; other.

Use `include_in_paper=yes` for a small set of representative, anonymized examples.
The automatic category is only a triage hint, not a human annotation.
"""
    (args.save_dir / "ERROR_ANALYSIS_GUIDE.md").write_text(taxonomy, encoding="utf-8")

    print("Reference:", args.reference_model)
    print("Strongest baseline:", baseline)
    print(paired_macro.to_string(index=False))
    print("Selected error cases:", selected_counts)
    print("Output:", args.save_dir)


if __name__ == "__main__":
    main()
