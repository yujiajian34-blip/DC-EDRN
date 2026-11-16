#!/usr/bin/env python
"""Build a frozen binary detection set from the official Microsoft BIPIA test files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd


DEFAULT_COMMIT = "a004b69ec0dd446e0afd461d98cb5e96e120a5d0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--positions", default="start,middle,end")
    parser.add_argument("--attacks-per-context", type=int, default=1)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_attacks(values):
    return [(f"{category}-{index}", text) for category, texts in values.items() for index, text in enumerate(texts)]


def context_text(sample, task):
    value = sample["context"]
    if isinstance(value, list):
        value = "\n".join(str(item) for item in value)
    question = sample.get("question")
    if question:
        return f"{value}\n\nQuestion: {question}"
    return str(value)


def insert(context, attack, position, rng):
    if position == "start":
        return f"{attack}\n{context}"
    if position == "end":
        return f"{context}\n{attack}"
    pivot = len(context) // 2
    nearby = [index for index in (context.rfind("\n", 0, pivot), context.find("\n", pivot)) if index >= 0]
    if nearby:
        pivot = min(nearby, key=lambda index: abs(index - pivot))
    return f"{context[:pivot]}\n{attack}\n{context[pivot:]}"


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    positions = [value.strip() for value in args.positions.split(",") if value.strip()]
    source_files = {
        "code_context": args.source_dir / "benchmark__code__test.jsonl",
        "email_context": args.source_dir / "benchmark__email__test.jsonl",
        "table_context": args.source_dir / "benchmark__table__test.jsonl",
        "code_attack": args.source_dir / "benchmark__code_attack_test.json",
        "text_attack": args.source_dir / "benchmark__text_attack_test.json",
    }
    missing = [str(path) for path in source_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing official BIPIA files: " + ", ".join(missing))
    code_attacks = flatten_attacks(load_json(source_files["code_attack"]))
    text_attacks = flatten_attacks(load_json(source_files["text_attack"]))
    tasks = [
        ("code", load_jsonl(source_files["code_context"]), code_attacks),
        ("email", load_jsonl(source_files["email_context"]), text_attacks),
        ("table", load_jsonl(source_files["table_context"]), text_attacks),
    ]
    records = []
    for task, contexts, attacks in tasks:
        for context_index, sample in enumerate(contexts):
            clean = context_text(sample, task)
            records.append({
                "id": f"bipia-{task}-{context_index:04d}-clean", "text": clean,
                "label": 0, "task": task, "position": "none", "attack_name": "none",
            })
            chosen = rng.sample(attacks, k=min(args.attacks_per_context, len(attacks)))
            for attack_index, (attack_name, attack_text) in enumerate(chosen):
                position = positions[(context_index + attack_index) % len(positions)]
                records.append({
                    "id": f"bipia-{task}-{context_index:04d}-attack-{attack_index}",
                    "text": insert(clean, attack_text, position, rng), "label": 1,
                    "task": task, "position": position, "attack_name": attack_name,
                })
    output = pd.DataFrame(records).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.out_file, index=False)
    summary = {
        "name": "BIPIA external prompt-injection detection adaptation",
        "upstream_repository": "https://github.com/microsoft/BIPIA",
        "upstream_commit": args.commit,
        "construction_seed": args.seed,
        "positions": positions,
        "attacks_per_context": args.attacks_per_context,
        "rows": len(output),
        "label_counts": {str(key): int(value) for key, value in output.label.value_counts().items()},
        "task_counts": {str(key): int(value) for key, value in output.task.value_counts().items()},
        "source_sha256": {name: sha256(path) for name, path in source_files.items()},
        "note": "Thresholds and model weights must remain frozen. This adaptation evaluates detection, not LLM attack success rate.",
    }
    args.out_file.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
