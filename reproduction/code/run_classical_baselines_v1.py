import argparse
import importlib.util
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "neuralchemy_context_gated_benchmark_v3.parquet"
DEFAULT_SAVE = ROOT / "results" / "classical_baselines_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Classical baselines under the V7 split and FPR protocol.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE)
    parser.add_argument("--models", default="WordLR,CharLR,CharSVM,HybridLR")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_v7_module():
    path = Path(__file__).with_name("DualResolutionCGEDRN-v7-search.py")
    spec = importlib.util.spec_from_file_location("drcgedrn_v7", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_vectorizer(name):
    if name == "WordLR":
        return TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000,
            sublinear_tf=True, strip_accents=None, dtype=np.float32,
        )
    if name in {"CharLR", "CharSVM"}:
        return TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100000,
            sublinear_tf=True, dtype=np.float32,
        )
    if name == "HybridLR":
        return FeatureUnion(
            [
                ("word", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=2,
                    max_features=40000, sublinear_tf=True, dtype=np.float32,
                )),
                ("char", TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 5), min_df=2,
                    max_features=60000, sublinear_tf=True, dtype=np.float32,
                )),
            ]
        )
    raise ValueError(f"Unknown baseline: {name}")


def build_classifier(name, seed):
    if name == "CharSVM":
        return LinearSVC(C=1.0, class_weight="balanced", random_state=seed, dual="auto")
    return LogisticRegression(
        C=4.0, class_weight="balanced", solver="liblinear",
        max_iter=2000, random_state=seed,
    )


def scores_to_probabilities(classifier, features):
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(features)[:, 1]
    return expit(classifier.decision_function(features))


def completed_keys(path):
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return set(zip(frame["Model"].astype(str), frame["Seed"].astype(int)))


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.save_dir / "models"
    model_dir.mkdir(exist_ok=True)
    v7 = load_v7_module()
    frame = v7.load_frame(args.data_file)
    train = frame[frame["split"].isin(["train", "train_context"])].reset_index(drop=True)
    eval_splits = [
        "val", "val_notinject", "val_overdefense", "diagnostic_test",
        "diagnostic_notinject", "diagnostic_obfuscation",
        "diagnostic_long_context", "diagnostic_overdefense",
    ]
    split_frames = {
        split: frame[frame["split"].eq(split)].reset_index(drop=True)
        for split in eval_splits
    }
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    run_path = args.save_dir / "baseline_runs.csv"
    metric_path = args.save_dir / "baseline_operating_points.csv"
    completed = completed_keys(run_path) if args.resume else set()

    for model_name in models:
        vectorizer = build_vectorizer(model_name)
        print(f"[VECTORIZER] {model_name}")
        train_features = vectorizer.fit_transform(train["text"].astype(str))
        eval_features = {
            split: vectorizer.transform(split_frame["text"].astype(str))
            for split, split_frame in split_frames.items()
        }
        feature_dim = int(train_features.shape[1])
        sample_weights = train.get("sample_weight", pd.Series(1.0, index=train.index)).to_numpy(float)

        for seed in seeds:
            if (model_name, seed) in completed:
                print(f"[RESUME SKIP] {model_name} seed={seed}")
                continue
            classifier = build_classifier(model_name, seed)
            start = time.perf_counter()
            classifier.fit(train_features, train["label"].to_numpy(int), sample_weight=sample_weights)
            fit_seconds = time.perf_counter() - start
            probabilities = {
                split: scores_to_probabilities(classifier, features)
                for split, features in eval_features.items()
            }
            labels = {
                split: split_frame["label"].to_numpy(int)
                for split, split_frame in split_frames.items()
            }
            val_labels, val_probs = labels["val"], probabilities["val"]
            ni_probs, od_probs = probabilities["val_notinject"], probabilities["val_overdefense"]
            f1_threshold = v7.optimal_threshold(val_labels, val_probs)
            threshold_args = SimpleNamespace(max_fpr_notinject=0.10, max_fpr_overdefense=0.15)
            safety_threshold, _, ni_fpr, od_fpr, feasible, selection_score = v7.constrained_threshold(
                val_labels, val_probs, ni_probs, od_probs, threshold_args
            )
            thresholds = {"F1Optimal": f1_threshold, "Safety10_15": safety_threshold}

            inference_start = time.perf_counter()
            diagnostic_text = split_frames["diagnostic_test"]["text"].astype(str)
            diagnostic_features = vectorizer.transform(diagnostic_text)
            _ = scores_to_probabilities(classifier, diagnostic_features)
            latency_ms = (time.perf_counter() - inference_start) * 1000 / max(1, len(diagnostic_text))

            trainable_params = int(np.asarray(classifier.coef_).size + np.asarray(classifier.intercept_).size)
            artifact_path = model_dir / f"{model_name}_seed{seed}.joblib"
            joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, artifact_path, compress=3)
            artifact_mb = artifact_path.stat().st_size / 1024**2
            run_row = {
                "Model": model_name, "Seed": seed, "FeatureDim": feature_dim,
                "TrainableParams": trainable_params, "ArtifactMB": artifact_mb,
                "FitSeconds": fit_seconds, "LatencyMsPerSample": latency_ms,
                "F1Threshold": f1_threshold, "SafetyThreshold": safety_threshold,
                "ValNotInjectFPR": ni_fpr, "ValOverdefenseFPR": od_fpr,
                "ThresholdFeasible": feasible, "SelectionScore": selection_score,
            }
            pd.DataFrame([run_row]).to_csv(
                run_path, mode="a", header=not run_path.exists(), index=False
            )

            rows = []
            for point_name, threshold in thresholds.items():
                for split in eval_splits:
                    metrics = v7.calculate_metrics(labels[split], probabilities[split], threshold)
                    rows.append(
                        {
                            "Model": model_name, "Seed": seed,
                            "OperatingPoint": point_name, "Split": split,
                            "Threshold": threshold, **metrics,
                        }
                    )
            pd.DataFrame(rows).to_csv(
                metric_path, mode="a", header=not metric_path.exists(), index=False
            )
            print(
                f"[DONE] {model_name} seed={seed} features={feature_dim:,} "
                f"safety_val_f1={selection_score:.4f} ni_fpr={ni_fpr:.4f} od_fpr={od_fpr:.4f}"
            )

    runs = pd.read_csv(run_path).drop_duplicates(["Model", "Seed"], keep="last")
    metrics = pd.read_csv(metric_path).drop_duplicates(
        ["Model", "Seed", "OperatingPoint", "Split"], keep="last"
    )
    runs.to_csv(run_path, index=False)
    metrics.to_csv(metric_path, index=False)
    summary = (
        metrics.groupby(["Model", "OperatingPoint", "Split"], as_index=False)
        .agg(
            Seeds=("Seed", "nunique"), F1Mean=("F1", "mean"), F1Std=("F1", "std"),
            PrecisionMean=("P", "mean"), RecallMean=("R", "mean"),
            FPRMean=("FPR", "mean"), FNRMean=("FNR", "mean"),
            PRAUCMean=("PRAUC", "mean"), MCCMean=("MCC", "mean"),
        )
    )
    summary.to_csv(args.save_dir / "baseline_summary.csv", index=False)
    print(f"Wrote baseline results to: {args.save_dir}")


if __name__ == "__main__":
    main()
