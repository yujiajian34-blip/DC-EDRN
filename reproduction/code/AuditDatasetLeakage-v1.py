#!/usr/bin/env python
"""Audit exact, group, template, and MinHash near-duplicate leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
SPACE = re.compile(r"\s+")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
HEX = re.compile(r"\b[0-9a-f]{16,}\b", re.I)
QUOTED = re.compile(r"(['\"]).{4,}?\1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--train-splits", default="train,train_context")
    parser.add_argument("--eval-splits", default="val,val_notinject,val_overdefense,diagnostic_test,diagnostic_notinject,diagnostic_obfuscation,diagnostic_long_context,diagnostic_overdefense")
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--num-permutations", type=int, default=32)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--lineage-columns", default="group_id,pair_id")
    parser.add_argument("--external-file", type=Path, default=None)
    parser.add_argument("--external-text-column", default="text")
    parser.add_argument("--external-label-column", default="label")
    parser.add_argument("--external-id-column", default="id")
    return parser.parse_args()


def normalize(text):
    text = ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", str(text)).casefold())
    return SPACE.sub(" ", text).strip()


def template(text):
    text = normalize(text)
    text = URL.sub("<url>", text)
    text = EMAIL.sub("<email>", text)
    text = HEX.sub("<hex>", text)
    text = NUMBER.sub("<num>", text)
    text = QUOTED.sub("<quoted>", text)
    return text


def shingles(text, width=5):
    compact = SPACE.sub(" ", normalize(text))
    if len(compact) <= width:
        return {compact}
    return {compact[index : index + width] for index in range(len(compact) - width + 1)}


def signature(values, permutations):
    if not values:
        return tuple([0] * permutations)
    prime = np.uint64(4294967311)
    base = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest(), "little")
            for value in values
        ),
        dtype=np.uint64,
    )
    rng = np.random.default_rng(20260812)
    coefficients = rng.integers(1, int(prime), size=permutations, dtype=np.uint64)
    offsets = rng.integers(0, int(prime), size=permutations, dtype=np.uint64)
    hashes = (coefficients[:, None] * base[None, :] + offsets[:, None]) % prime
    return tuple(int(value) for value in hashes.min(axis=1))


def jaccard(left, right):
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def preview(text, limit=240):
    value = SPACE.sub(" ", str(text)).strip()
    return value[:limit]


def read_external(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    raise ValueError(f"Unsupported external format: {path}")


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.data_file)
    frame = frame.dropna(subset=["text", "split"]).copy().reset_index(drop=True)
    frame["normalized"] = frame["text"].map(normalize)
    frame["template"] = frame["text"].map(template)
    train_splits = {value.strip() for value in args.train_splits.split(",") if value.strip()}
    eval_splits = {value.strip() for value in args.eval_splits.split(",") if value.strip()}
    train = frame[frame.split.isin(train_splits)].copy()
    evaluation = frame[frame.split.isin(eval_splits)].copy()
    if args.external_file is not None:
        external = read_external(args.external_file)
        if args.external_text_column not in external:
            raise ValueError(f"External text column not found: {args.external_text_column}")
        external_labels = external[args.external_label_column] if args.external_label_column in external else np.nan
        external_ids = (
            external[args.external_id_column].astype(str)
            if args.external_id_column in external
            else pd.Series([f"external_{index}" for index in range(len(external))])
        )
        external = pd.DataFrame({
            "id": external_ids,
            "text": external[args.external_text_column].astype(str),
            "label": external_labels,
            "split": "external_test",
        })
        external["normalized"] = external["text"].map(normalize)
        external["template"] = external["text"].map(template)
        evaluation = pd.concat([evaluation, external], ignore_index=True, sort=False)

    findings = []
    for kind, column in [("normalized_exact", "normalized"), ("template_exact", "template")]:
        lookup = train.groupby(column).head(1).set_index(column)
        matched = evaluation[evaluation[column].isin(lookup.index)]
        for row in matched.itertuples():
            source = lookup.loc[getattr(row, column)]
            findings.append({
                "kind": kind, "similarity": 1.0, "eval_split": row.split,
                "eval_id": getattr(row, "id", row.Index), "train_id": source.get("id", source.name),
                "eval_label": getattr(row, "label", np.nan), "train_label": source.get("label", np.nan),
                "eval_text_preview": preview(row.text), "train_text_preview": preview(source.get("text", "")),
            })

    lineage_columns = [value.strip() for value in args.lineage_columns.split(",") if value.strip()]
    for column in lineage_columns:
        if column not in frame.columns:
            continue
        train_ids = train[column].astype("string").str.strip()
        eval_ids = evaluation[column].astype("string").str.strip()
        invalid = {"", "nan", "none", "null", "<na>"}
        train_values = set(train_ids[train_ids.notna() & ~train_ids.str.casefold().isin(invalid)])
        valid_eval = eval_ids.notna() & ~eval_ids.str.casefold().isin(invalid)
        overlap = evaluation[valid_eval & eval_ids.isin(train_values)]
        for row in overlap.itertuples():
            findings.append({
                "kind": f"{column}_overlap", "similarity": 1.0, "eval_split": row.split,
                "eval_id": getattr(row, "id", row.Index), "train_id": str(getattr(row, column)),
                "eval_label": getattr(row, "label", np.nan), "train_label": np.nan,
                "eval_text_preview": preview(row.text), "train_text_preview": "",
            })

    permutations = args.num_permutations
    if permutations % args.bands:
        raise ValueError("--num-permutations must be divisible by --bands")
    rows_per_band = permutations // args.bands
    train_sets = [shingles(text) for text in train.text]
    eval_sets = [shingles(text) for text in evaluation.text]
    buckets = defaultdict(list)
    for index, values in enumerate(train_sets):
        sig = signature(values, permutations)
        for band in range(args.bands):
            block = sig[band * rows_per_band : (band + 1) * rows_per_band]
            buckets[(band, block)].append(index)
    for eval_index, values in enumerate(eval_sets):
        sig = signature(values, permutations)
        candidates = set()
        for band in range(args.bands):
            block = sig[band * rows_per_band : (band + 1) * rows_per_band]
            candidates.update(buckets.get((band, block), []))
        best = None
        for train_index in candidates:
            score = jaccard(values, train_sets[train_index])
            if best is None or score > best[0]:
                best = (score, train_index)
        if best and args.near_threshold <= best[0] < 1.0:
            eval_row = evaluation.iloc[eval_index]
            train_row = train.iloc[best[1]]
            findings.append({
                "kind": "char5_near_duplicate", "similarity": best[0], "eval_split": eval_row.split,
                "eval_id": eval_row.get("id", eval_row.name), "train_id": train_row.get("id", train_row.name),
                "eval_label": eval_row.get("label", np.nan), "train_label": train_row.get("label", np.nan),
                "eval_text_preview": preview(eval_row.text), "train_text_preview": preview(train_row.text),
            })

    result = pd.DataFrame(findings)
    result.to_csv(args.save_dir / "leakage_findings.csv", index=False)
    summary = {
        "data_file": str(args.data_file), "external_file": str(args.external_file) if args.external_file else None,
        "rows": len(frame), "train_rows": len(train),
        "evaluation_rows": len(evaluation), "near_threshold": args.near_threshold,
        "finding_counts": result["kind"].value_counts().to_dict() if not result.empty else {},
        "split_counts": result["eval_split"].value_counts().to_dict() if not result.empty else {},
    }
    (args.save_dir / "leakage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
