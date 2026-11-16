#!/usr/bin/env python
"""Convert the public PIGuard BIPIA attack-only test files to a frozen detection set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--commit", default="1b5751e88bf7475acbedfc8eda795ce060307c84")
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = parse_args()
    records = []
    hashes = {}
    for subset, filename in [("text", "BIPIA_text.json"), ("code", "BIPIA_code.json")]:
        path = args.source_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[filename] = sha256(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for category, texts in data.items():
            for index, value in enumerate(texts):
                records.append({
                    "id": f"piguard-bipia-{subset}-{len(records):04d}", "text": str(value),
                    "label": 1, "subset": subset, "category": category, "source_index": index,
                })
    output = pd.DataFrame(records)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.out_file, index=False)
    summary = {
        "name": "PIGuard BIPIA attack-only test",
        "upstream_repository": "https://github.com/leolee99/PIGuard",
        "upstream_commit": args.commit,
        "rows": len(output),
        "subset_counts": {str(key): int(value) for key, value in output.subset.value_counts().items()},
        "source_sha256": hashes,
        "note": "Attack-only set. Report recall/attack detection accuracy, not F1 or FPR.",
    }
    args.out_file.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
