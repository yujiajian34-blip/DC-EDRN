#!/usr/bin/env python
"""Evaluate frozen local checkpoints and public guards on an untouched dataset."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--checkpoint-dirs", required=True, help="Comma-separated directories searched recursively for .pth files.")
    parser.add_argument("--public-model-root", type=Path, default=None)
    parser.add_argument("--public-threshold-results", type=Path, default=None)
    parser.add_argument("--public-operating-point", default="Safety10_15")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--source-split", default=None, help="Optional split value to select before evaluation.")
    parser.add_argument("--positive-labels", default="1,injection,jailbreak,attack,malicious,unsafe")
    parser.add_argument("--group-columns", default="task,position,subset,category")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-words", type=int, default=128)
    parser.add_argument("--word-vocab", type=int, default=8192)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def import_module(filename, name):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V7 = import_module("DualResolutionCGEDRN-v7-search.py", "external_v7")
MATCHED = import_module("MatchedLocalEncoderBaselines-v1.py", "external_matched")


def read_frame(path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported dataset format: {path}")


def prepare_frame(frame, args):
    if args.text_column not in frame or args.label_column not in frame:
        raise ValueError(f"Dataset needs columns {args.text_column!r} and {args.label_column!r}")
    positive = {value.strip().casefold() for value in args.positive_labels.split(",") if value.strip()}
    labels = frame[args.label_column]
    if pd.api.types.is_numeric_dtype(labels):
        mapped = labels.astype(int)
    else:
        mapped = labels.astype(str).str.strip().str.casefold().isin(positive).astype(int)
    output = pd.DataFrame({
        "id": frame[args.id_column].astype(str) if args.id_column in frame else [f"external_{index}" for index in range(len(frame))],
        "text": frame[args.text_column].astype(str),
        "label": mapped,
        "split": "external_test",
        "span_start": -1,
        "span_end": -1,
        "duration_bucket": "none",
        "sample_weight": 1.0,
        "intent_label": np.where(mapped.to_numpy() > 0, 2, 0),
    })
    for column in [value.strip() for value in args.group_columns.split(",") if value.strip()]:
        if column in frame:
            output[column] = frame[column].astype(str).to_numpy()
    if set(output.label.unique()) - {0, 1}:
        raise ValueError("Labels must map to binary 0/1")
    return output


def metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    mixed = len(np.unique(labels)) == 2
    return {
        "N": len(labels), "Threshold": threshold,
        "F1": f1_score(labels, predictions, zero_division=0),
        "P": precision_score(labels, predictions, zero_division=0),
        "R": recall_score(labels, predictions, zero_division=0),
        "Acc": accuracy_score(labels, predictions),
        "FPR": fp / max(1, fp + tn), "FNR": fn / max(1, fn + tp),
        "MCC": matthews_corrcoef(labels, predictions) if mixed else math.nan,
        "PRAUC": average_precision_score(labels, scores) if mixed else math.nan,
        "ROCAUC": roc_auc_score(labels, scores) if mixed else math.nan,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


@torch.inference_mode()
def local_scores(model, dataset, device, batch_size, amp):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    values = []
    model.eval()
    for batch in tqdm(loader, desc="Frozen local inference", leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(batch[0].to(device), batch[6].to(device))["logits"]
        values.extend(torch.sigmoid(output.float()).cpu().tolist())
    return np.asarray(values, dtype=float)


def public_thresholds(path, point):
    if path is None:
        return {}
    frame = pd.read_csv(path)
    selected = frame[frame.OperatingPoint.eq(point)]
    return selected.groupby("Model")["Threshold"].first().to_dict()


@torch.inference_mode()
def public_scores(model_id, path, texts, device, batch_size):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    trust = model_id.lower() == "leolee99/piguard"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=trust, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=trust, torch_dtype=torch.float32).to(device).eval()
    values = []
    for start in tqdm(range(0, len(texts), batch_size), desc=model_id, leave=False):
        encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        logits = model(**encoded).logits.float()
        values.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
    del model
    return np.asarray(values, dtype=float)


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    raw_frame = read_frame(args.data_file)
    if args.source_split is not None:
        if "split" not in raw_frame:
            raise ValueError("--source-split was provided but the dataset has no 'split' column")
        raw_frame = raw_frame[raw_frame["split"].astype(str).eq(args.source_split)].copy()
        if raw_frame.empty:
            raise ValueError(f"No rows found for --source-split {args.source_split!r}")
    frame = prepare_frame(raw_frame, args)
    dataset = V7.encode_frame(frame, 512, args.max_words, args.word_vocab)
    labels = frame.label.to_numpy(dtype=int)
    prediction_rows, metric_rows = [], []
    checkpoints = []
    for directory in [Path(value.strip()) for value in args.checkpoint_dirs.split(",") if value.strip()]:
        checkpoints.extend(sorted(directory.rglob("*.pth")))
    if not checkpoints:
        raise FileNotFoundError("No local checkpoints were found")
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = MATCHED.MatchedLocalEncoderModel(config, int(checkpoint["downsample"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        score = local_scores(model, dataset, device, args.batch_size, bool(args.amp and device.type == "cuda"))
        threshold = float(checkpoint["threshold"])
        name = str(config["name"])
        seed = int(checkpoint["seed"])
        metric_rows.append({"Model": name, "Seed": seed, "Source": "frozen_local", **metrics(labels, score, threshold)})
        for index, (sample_id, label, value) in enumerate(zip(frame.id, labels, score)):
            row = {"id": sample_id, "label": int(label), "Model": name, "Seed": seed, "score": float(value), "threshold": threshold, "prediction": int(value >= threshold)}
            for column in [item.strip() for item in args.group_columns.split(",") if item.strip() and item in frame]:
                row[column] = frame.iloc[index][column]
            prediction_rows.append(row)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    thresholds = public_thresholds(args.public_threshold_results, args.public_operating_point)
    if args.public_model_root is not None:
        for model_id in ["protectai/deberta-v3-base-prompt-injection-v2", "leolee99/PIGuard"]:
            path = args.public_model_root / re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id)
            if not path.exists():
                continue
            score = public_scores(model_id, path, frame.text.tolist(), device, min(args.batch_size, 32))
            threshold = float(thresholds.get(model_id, 0.0))
            metric_rows.append({"Model": model_id, "Seed": 0, "Source": f"public_{args.public_operating_point}", **metrics(labels, score, threshold)})
            for index, (sample_id, label, value) in enumerate(zip(frame.id, labels, score)):
                row = {"id": sample_id, "label": int(label), "Model": model_id, "Seed": 0, "score": float(value), "threshold": threshold, "prediction": int(value >= threshold)}
                for column in [item.strip() for item in args.group_columns.split(",") if item.strip() and item in frame]:
                    row[column] = frame.iloc[index][column]
                prediction_rows.append(row)
            if device.type == "cuda": torch.cuda.empty_cache()
    predictions = pd.DataFrame(prediction_rows)
    results = pd.DataFrame(metric_rows)
    predictions.to_parquet(args.save_dir / "external_predictions.parquet", index=False)
    results.to_csv(args.save_dir / "external_metrics.csv", index=False)
    subgroup_rows = []
    group_columns = [value.strip() for value in args.group_columns.split(",") if value.strip() and value.strip() in predictions]
    for column in group_columns:
        for (model, seed, group), part in predictions.groupby(["Model", "Seed", column], dropna=False):
            subgroup_rows.append({"GroupColumn": column, "Group": group, "Model": model, "Seed": int(seed), **metrics(part.label, part.score, float(part.threshold.iloc[0]))})
    pd.DataFrame(subgroup_rows).to_csv(args.save_dir / "external_subgroup_metrics.csv", index=False)
    summary = results.groupby(["Model", "Source"], as_index=False).agg(Seeds=("Seed", "nunique"), F1Mean=("F1", "mean"), F1Std=("F1", "std"), PrecisionMean=("P", "mean"), RecallMean=("R", "mean"), FPRMean=("FPR", "mean"), PRAUCMean=("PRAUC", "mean"), MCCMean=("MCC", "mean"))
    summary.to_csv(args.save_dir / "external_summary.csv", index=False)
    (args.save_dir / "frozen_protocol.json").write_text(json.dumps({"data_file": str(args.data_file), "checkpoint_count": len(checkpoints), "public_operating_point": args.public_operating_point, "rows": len(frame), "label_counts": {str(key): int(value) for key, value in frame.label.value_counts().items()}, "single_class_reporting_note": "For all-positive sets, report Recall (attack detection accuracy); F1/FPR/PRAUC are not comparable." if frame.label.nunique() == 1 else None}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
