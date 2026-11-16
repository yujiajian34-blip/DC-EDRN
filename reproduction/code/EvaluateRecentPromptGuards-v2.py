#!/usr/bin/env python
"""Evaluate public prompt guards using non-saturating logit-margin scores.

The script performs zero-shot evaluation only.  It calibrates thresholds on
the V3 validation splits and never updates model weights.  Per-example scores
are cached so interrupted runs can resume without repeating inference.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


DEFAULT_MODELS = [
    "protectai/deberta-v3-base-prompt-injection-v2",
    "leolee99/PIGuard",
    "meta-llama/Llama-Prompt-Guard-2-22M",
    "meta-llama/Llama-Prompt-Guard-2-86M",
]
CALIBRATION_SPLITS = ["val", "val_notinject", "val_overdefense"]
EVALUATION_SPLITS = [
    "diagnostic_test",
    "diagnostic_notinject",
    "diagnostic_obfuscation",
    "diagnostic_long_context",
    "diagnostic_overdefense",
]
POSITIVE_HINTS = ("malicious", "injection", "jailbreak", "attack", "unsafe")
NEGATIVE_HINTS = ("benign", "safe", "clean", "normal", "no_injection", "legitimate")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(r"/data/neuralchemy_context_gated_benchmark_v3.parquet"),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(r"/results/recent_prompt_guards_v1"),
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Optional root containing offline model snapshots named with safe_slug(model_id).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require local model snapshots and never contact Hugging Face.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-fpr-notinject", type=float, default=0.10)
    parser.add_argument("--max-fpr-overdefense", type=float, default=0.15)
    parser.add_argument("--low-fpr-targets", default="0.001,0.005,0.01,0.05")
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def safe_slug(model_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id)


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def choose_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(choice)


def resolve_positive_indices(config):
    id2label = {int(key): str(value).lower() for key, value in config.id2label.items()}
    positive = [
        index
        for index, label in id2label.items()
        if any(hint in label for hint in POSITIVE_HINTS)
        and not any(hint in label for hint in NEGATIVE_HINTS)
    ]
    negative = [
        index for index, label in id2label.items() if any(hint in label for hint in NEGATIVE_HINTS)
    ]
    if positive:
        return sorted(set(positive)), id2label, "label_heuristic"
    if len(id2label) == 2 and negative:
        return [index for index in sorted(id2label) if index not in negative], id2label, "negative_complement"
    if len(id2label) == 2:
        return [1], id2label, "binary_default_index_1"
    raise ValueError(f"Cannot infer malicious class from id2label={id2label}")


def load_model(model_id, device, model_root=None, offline=False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_path = model_root / safe_slug(model_id) if model_root is not None else None
    if local_path is not None and local_path.exists():
        model_source = str(local_path)
        revision = "offline_snapshot"
        local_files_only = True
    elif offline:
        raise FileNotFoundError(
            f"Offline snapshot missing for {model_id}: expected {local_path}"
        )
    else:
        from huggingface_hub import model_info

        info = model_info(model_id)
        revision = info.sha
        model_source = model_id
        local_files_only = False
    trust_remote_code = model_id.lower() == "leolee99/piguard"
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=None if local_files_only else revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        revision=None if local_files_only else revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch.float32,
    )
    model.to(device).eval()
    positive_indices, labels, mapping_method = resolve_positive_indices(model.config)
    return tokenizer, model, revision, positive_indices, labels, mapping_method


@torch.inference_mode()
def score_texts(texts, tokenizer, model, positive_indices, batch_size, max_length, device):
    scores = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Inference", leave=False):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits.float()
        negative_indices = [index for index in range(logits.shape[-1]) if index not in positive_indices]
        if len(positive_indices) == 1 and len(negative_indices) == 1:
            batch_scores = logits[:, positive_indices[0]] - logits[:, negative_indices[0]]
        else:
            positive_score = torch.logsumexp(logits[:, positive_indices], dim=-1)
            negative_score = torch.logsumexp(logits[:, negative_indices], dim=-1)
            batch_scores = positive_score - negative_score
        scores.append(batch_scores.cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def metric_row(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "N": int(len(labels)),
        "PositiveRate": float(labels.mean()) if len(labels) else math.nan,
        "Threshold": float(threshold),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "P": float(precision_score(labels, predictions, zero_division=0)),
        "R": float(recall_score(labels, predictions, zero_division=0)),
        "Acc": float((predictions == labels).mean()),
        "FPR": float(fp / max(1, fp + tn)),
        "FNR": float(fn / max(1, fn + tp)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "PRAUC": float(average_precision_score(labels, probabilities)) if labels.min() != labels.max() else math.nan,
        "ROCAUC": float(roc_auc_score(labels, probabilities)) if labels.min() != labels.max() else math.nan,
    }


def threshold_candidates(*arrays):
    values = np.unique(np.concatenate([np.asarray(array, dtype=np.float64) for array in arrays]))
    return np.concatenate(
        [
            [np.nextafter(values.max(), np.inf)],
            values[::-1],
            [np.nextafter(values.min(), -np.inf)],
        ]
    )


def f1_optimal_threshold(labels, probabilities):
    best = None
    for threshold in threshold_candidates(probabilities):
        metrics = metric_row(labels, probabilities, threshold)
        key = (metrics["F1"], metrics["P"], metrics["R"], threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def constrained_threshold(val_labels, val_probs, notinject_probs, overdefense_probs, max_ni, max_od):
    best = None
    fallback = None
    for threshold in threshold_candidates(val_probs, notinject_probs, overdefense_probs):
        val_metrics = metric_row(val_labels, val_probs, threshold)
        ni_fpr = float((notinject_probs >= threshold).mean())
        od_fpr = float((overdefense_probs >= threshold).mean())
        feasible = ni_fpr <= max_ni + 1e-12 and od_fpr <= max_od + 1e-12
        feasible_key = (val_metrics["F1"], val_metrics["P"], val_metrics["R"], threshold)
        violation = max(0.0, ni_fpr - max_ni) + max(0.0, od_fpr - max_od)
        fallback_key = (-violation, val_metrics["F1"], threshold)
        if feasible and (best is None or feasible_key > best[0]):
            best = (feasible_key, float(threshold))
        if fallback is None or fallback_key > fallback[0]:
            fallback = (fallback_key, float(threshold))
    return (best or fallback)[1]


def low_fpr_threshold(benign_probabilities, target_fpr):
    benign_probabilities = np.asarray(benign_probabilities, dtype=np.float64)
    feasible = []
    for threshold in threshold_candidates(benign_probabilities):
        if float((benign_probabilities >= threshold).mean()) <= target_fpr + 1e-12:
            feasible.append(float(threshold))
    return min(feasible) if feasible else float(np.nextafter(benign_probabilities.max(), np.inf))


@torch.inference_mode()
def measure_latency(text, tokenizer, model, max_length, device, repeats):
    def one_call():
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        _ = model(**encoded).logits
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(5):
        one_call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        one_call()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples)), float(np.percentile(samples, 95))


def evaluate_model(model_id, frame, args, device):
    slug = safe_slug(model_id)
    cache_path = args.save_dir / "predictions" / f"{slug}.parquet"
    metadata_path = args.save_dir / "predictions" / f"{slug}.json"
    required_splits = CALIBRATION_SPLITS + EVALUATION_SPLITS
    selected = frame[frame["split"].isin(required_splits)].copy().reset_index(drop=True)

    tokenizer = model = None
    metadata = {}
    if args.resume and cache_path.exists() and metadata_path.exists():
        predictions = pd.read_parquet(cache_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"[RESUME] Loaded cached scores for {model_id}")
    else:
        tokenizer, model, revision, positive_indices, labels, mapping_method = load_model(
            model_id, device, args.model_root, args.offline
        )
        scores = score_texts(
            selected["text"].astype(str).tolist(),
            tokenizer,
            model,
            positive_indices,
            args.batch_size,
            args.max_length,
            device,
        )
        predictions = selected[["id", "split", "label"]].copy()
        predictions["score"] = scores
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(cache_path, index=False)
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        latency_median, latency_p95 = measure_latency(
            selected.iloc[0]["text"],
            tokenizer,
            model,
            args.max_length,
            device,
            args.latency_repeats,
        )
        metadata = {
            "model_id": model_id,
            "revision": revision,
            "id2label": labels,
            "positive_indices": positive_indices,
            "mapping_method": mapping_method,
            "parameters": parameter_count,
            "fp32_size_mb": parameter_count * 4 / 1024**2,
            "latency_batch1_median_ms": latency_median,
            "latency_batch1_p95_ms": latency_p95,
            "device": str(device),
            "max_length": args.max_length,
            "score_type": "positive_logit_minus_negative_logit",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    split_data = {
        split: predictions[predictions["split"].eq(split)].reset_index(drop=True)
        for split in required_splits
    }
    val = split_data["val"]
    notinject = split_data["val_notinject"]
    overdefense = split_data["val_overdefense"]
    val_labels = val["label"].to_numpy(dtype=np.int64)
    score_column = "score"
    val_probs = val[score_column].to_numpy(dtype=np.float64)
    ni_probs = notinject[score_column].to_numpy(dtype=np.float64)
    od_probs = overdefense[score_column].to_numpy(dtype=np.float64)
    benign_calibration = np.concatenate(
        [val_probs[val_labels == 0], ni_probs, od_probs]
    )

    operating_points = {
        "ReleasedDefault": 0.0,
        "F1Optimal": f1_optimal_threshold(val_labels, val_probs),
        "Safety10_15": constrained_threshold(
            val_labels,
            val_probs,
            ni_probs,
            od_probs,
            args.max_fpr_notinject,
            args.max_fpr_overdefense,
        ),
    }
    for target in [float(value) for value in args.low_fpr_targets.split(",") if value.strip()]:
        operating_points[f"TPRatFPR{100 * target:g}pct"] = low_fpr_threshold(
            benign_calibration, target
        )

    output_path = args.save_dir / "recent_guard_results.csv"
    for point_name, threshold in operating_points.items():
        calibration_metrics = metric_row(val_labels, val_probs, threshold)
        calibration_ni_fpr = float((ni_probs >= threshold).mean())
        calibration_od_fpr = float((od_probs >= threshold).mean())
        for split in EVALUATION_SPLITS:
            part = split_data[split]
            metrics = metric_row(
                part["label"].to_numpy(dtype=np.int64),
                part[score_column].to_numpy(dtype=np.float64),
                threshold,
            )
            row = {
                "Model": model_id,
                "Revision": metadata.get("revision", ""),
                "OperatingPoint": point_name,
                "Split": split,
                "Parameters": metadata.get("parameters", math.nan),
                "FP32SizeMB": metadata.get("fp32_size_mb", math.nan),
                "LatencyBatch1MedianMS": metadata.get("latency_batch1_median_ms", math.nan),
                "LatencyBatch1P95MS": metadata.get("latency_batch1_p95_ms", math.nan),
                "CalibrationValF1": calibration_metrics["F1"],
                "CalibrationValFPR": calibration_metrics["FPR"],
                "CalibrationNotInjectFPR": calibration_ni_fpr,
                "CalibrationOverdefenseFPR": calibration_od_fpr,
                **metrics,
            }
            append_csv(output_path, row)

    del tokenizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    frame = pd.read_parquet(args.data_file)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    print(f"Device: {device}")
    print(f"Data: {args.data_file}")
    print(f"Models: {models}")
    print(f"Rows: {frame[frame['split'].isin(CALIBRATION_SPLITS + EVALUATION_SPLITS)]['split'].value_counts().to_dict()}")

    error_path = args.save_dir / "errors.csv"
    result_path = args.save_dir / "recent_guard_results.csv"
    point_count = 2 + len([value for value in args.low_fpr_targets.split(",") if value.strip()])
    expected_rows = point_count * len(EVALUATION_SPLITS)
    for model_id in models:
        if args.resume and result_path.exists():
            existing = pd.read_csv(result_path)
            model_rows = existing[existing["Model"].eq(model_id)]
            if len(model_rows) >= expected_rows:
                print(f"[RESUME SKIP] {model_id}")
                continue
            if len(model_rows):
                existing = existing[~existing["Model"].eq(model_id)]
                if len(existing):
                    existing.to_csv(result_path, index=False)
                else:
                    result_path.unlink()
        print(f"[START] {model_id}")
        try:
            evaluate_model(model_id, frame, args, device)
            print(f"[DONE] {model_id}")
        except Exception as error:
            append_csv(
                error_path,
                {
                    "Model": model_id,
                    "ErrorType": type(error).__name__,
                    "Error": str(error),
                    "Traceback": traceback.format_exc(),
                },
            )
            print(f"[FAILED] {model_id}: {type(error).__name__}: {error}")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
