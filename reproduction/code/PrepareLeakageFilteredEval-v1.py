#!/usr/bin/env python
"""Create a fixed internal sensitivity-test set with train-like rows removed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--findings-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--splits", default="diagnostic_test")
    parser.add_argument(
        "--exclude-kinds",
        default="normalized_exact,template_exact,char5_near_duplicate,group_id_overlap,pair_id_overlap",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_parquet(args.data_file)
    findings = pd.read_csv(args.findings_file)
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    kinds = {value.strip() for value in args.exclude_kinds.split(",") if value.strip()}
    selected = frame[frame["split"].astype(str).isin(splits)].copy()
    flagged = findings[
        findings["kind"].astype(str).isin(kinds)
        & findings["eval_split"].astype(str).isin(splits)
    ]
    excluded_ids = set(flagged["eval_id"].astype(str))
    clean = selected[~selected["id"].astype(str).isin(excluded_ids)].copy()
    if clean.empty:
        raise ValueError("Filtering removed every selected row")
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(args.out_file, index=False)
    summary = {
        "data_file": str(args.data_file),
        "findings_file": str(args.findings_file),
        "splits": sorted(splits),
        "exclude_kinds": sorted(kinds),
        "rows_before": int(len(selected)),
        "unique_rows_excluded": int(len(selected) - len(clean)),
        "rows_after": int(len(clean)),
        "label_counts_after": {str(key): int(value) for key, value in clean["label"].value_counts().items()},
        "finding_counts": {str(key): int(value) for key, value in flagged["kind"].value_counts().items()},
    }
    summary_path = args.out_file.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
