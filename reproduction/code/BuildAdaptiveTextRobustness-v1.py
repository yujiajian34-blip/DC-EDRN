#!/usr/bin/env python
"""Build deterministic text-space robustness suites from a frozen test set."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_FAMILIES = "case_flip,typo,separator,leet,homoglyph,zero_width,fullwidth,punctuation,combined"
SEVERITIES = {"low": 0.03, "medium": 0.07, "high": 0.12}
LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})
HOMOGLYPH = str.maketrans({
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "A": "А", "C": "С", "E": "Е", "O": "О", "P": "Р", "X": "Х",
})
REVERSE_HOMOGLYPH = str.maketrans({
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y",
    "А": "A", "С": "C", "Е": "E", "О": "O", "Р": "P", "Х": "X",
})
PUNCT = [".", "-", "_", "/", "|", ":"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--families", default=DEFAULT_FAMILIES)
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--max-base-rows", type=int, default=0)
    p.add_argument("--emit-canonicalized", action="store_true")
    return p.parse_args()


def stable_rng(seed: int, *parts: object) -> random.Random:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def choose_positions(text: str, rate: float, rng: random.Random, predicate=str.isalpha):
    positions = [i for i, char in enumerate(text) if predicate(char)]
    if not positions:
        return []
    count = max(1, min(len(positions), round(len(positions) * rate)))
    return sorted(rng.sample(positions, count))


def case_flip(text, rate, rng):
    chars = list(text)
    for i in choose_positions(text, rate, rng):
        chars[i] = chars[i].swapcase()
    return "".join(chars)


def typo(text, rate, rng):
    chars = list(text)
    positions = choose_positions(text, rate, rng)
    for i in reversed(positions):
        if i >= len(chars) or not chars[i].isalpha():
            continue
        action = rng.choice(["delete", "duplicate", "swap"])
        if action == "delete" and len(chars) > 2:
            del chars[i]
        elif action == "duplicate":
            chars.insert(i, chars[i])
        elif i + 1 < len(chars) and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def separator(text, rate, rng):
    matches = list(re.finditer(r"[A-Za-z]{5,}", text))
    if not matches:
        return punctuation(text, rate, rng)
    selected = rng.sample(matches, max(1, min(len(matches), round(len(matches) * max(rate * 3, .15)))))
    result = text
    for match in reversed(selected):
        word = match.group(0)
        step = max(2, round(1 / max(rate, .03)))
        sep = rng.choice([" ", ".", "-", "_"])
        changed = sep.join(word[i:i + step] for i in range(0, len(word), step))
        result = result[:match.start()] + changed + result[match.end():]
    return result


def mapped(text, rate, rng, table):
    eligible = [i for i, char in enumerate(text) if ord(char) in table]
    if not eligible:
        return case_flip(text, rate, rng)
    count = max(1, min(len(eligible), round(len(eligible) * rate * 2)))
    chosen = set(rng.sample(eligible, count))
    output = []
    for i, char in enumerate(text):
        if i not in chosen:
            output.append(char)
            continue
        value = table.get(ord(char), char)
        output.append(chr(value) if isinstance(value, int) else value)
    return "".join(output)


def leet(text, rate, rng):
    return mapped(text, rate, rng, LEET)


def homoglyph(text, rate, rng):
    return mapped(text, rate, rng, HOMOGLYPH)


def zero_width(text, rate, rng):
    positions = set(choose_positions(text, rate, rng))
    return "".join(char + ("\u200b" if i in positions else "") for i, char in enumerate(text))


def fullwidth(text, rate, rng):
    positions = set(choose_positions(text, rate, rng, lambda c: 0x21 <= ord(c) <= 0x7E))
    return "".join(chr(ord(char) + 0xFEE0) if i in positions else char for i, char in enumerate(text))


def punctuation(text, rate, rng):
    positions = set(choose_positions(text, rate, rng))
    return "".join(char + (rng.choice(PUNCT) if i in positions else "") for i, char in enumerate(text))


def combined(text, rate, rng):
    value = zero_width(text, rate * .55, rng)
    value = leet(value, rate * .75, rng)
    value = case_flip(value, rate * .55, rng)
    value = typo(value, rate * .45, rng)
    return value


TRANSFORMS = {
    "case_flip": case_flip, "typo": typo, "separator": separator, "leet": leet,
    "homoglyph": homoglyph, "zero_width": zero_width, "fullwidth": fullwidth,
    "punctuation": punctuation, "combined": combined,
}


def change_ratio(before: str, after: str) -> float:
    return 1.0 - difflib.SequenceMatcher(None, before, after, autojunk=False).ratio()


def canonicalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = "".join(char for char in value if unicodedata.category(char) != "Cf")
    value = value.translate(REVERSE_HOMOGLYPH)
    value = re.sub(r"(?<=[A-Za-z])[._|:/-](?=[A-Za-z])", "", value)
    spaced = re.compile(r"\b(?:[A-Za-z][\s._|:/-]+){2,}[A-Za-z]\b")
    value = spaced.sub(lambda match: re.sub(r"[\s._|:/-]+", "", match.group(0)), value)
    reverse_leet = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t"}
    value = re.sub(
        r"(?<=[A-Za-z])[431057](?=[A-Za-z])",
        lambda match: reverse_leet[match.group(0)],
        value,
    )
    return value


def write_variant(frame: pd.DataFrame, path: Path, preprocessing: str):
    output = frame.copy()
    output["preprocessing"] = preprocessing
    if preprocessing == "canonicalized":
        output["text"] = output["text"].map(canonicalize)
        output["id"] = output["id"].astype(str) + "::canonicalized"
    output.to_parquet(path, index=False)


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.source_file)
    required = {"id", "text", "label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Source data is missing columns: {sorted(missing)}")
    if args.max_base_rows > 0:
        frame = frame.sample(min(args.max_base_rows, len(frame)), random_state=args.seed).reset_index(drop=True)
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    unknown = sorted(set(families) - set(TRANSFORMS))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}")

    base_cols = [column for column in ["id", "text", "label", "source", "category"] if column in frame]
    clean = frame[base_cols].copy()
    clean["base_id"] = clean["id"].astype(str)
    clean["id"] = clean.base_id + "::clean"
    clean["attack_family"] = "clean"
    clean["severity"] = "clean"
    clean["replicate"] = 0
    clean["change_ratio"] = 0.0
    write_variant(clean, args.output_dir / "clean.parquet", "raw")
    if args.emit_canonicalized:
        write_variant(clean, args.output_dir / "clean_canonicalized.parquet", "canonicalized")

    counts = {"clean": len(clean)}
    if args.emit_canonicalized:
        counts["clean_canonicalized"] = len(clean)
    for family in families:
        rows = []
        transform = TRANSFORMS[family]
        for record in frame[base_cols].itertuples(index=False):
            values = record._asdict()
            base_id, text = str(values["id"]), unicodedata.normalize("NFC", str(values["text"]))
            for severity, rate in SEVERITIES.items():
                for replicate in range(args.replicates):
                    rng = stable_rng(args.seed, base_id, family, severity, replicate)
                    changed = transform(text, rate, rng)
                    row = dict(values)
                    row.update({
                        "base_id": base_id,
                        "id": f"{base_id}::{family}::{severity}::{replicate}",
                        "text": changed,
                        "attack_family": family,
                        "severity": severity,
                        "replicate": replicate,
                        "change_ratio": change_ratio(text, changed),
                    })
                    rows.append(row)
        output = pd.DataFrame(rows)
        write_variant(output, args.output_dir / f"{family}.parquet", "raw")
        if args.emit_canonicalized:
            write_variant(output, args.output_dir / f"{family}_canonicalized.parquet", "canonicalized")
        counts[family] = len(output)
        if args.emit_canonicalized:
            counts[f"{family}_canonicalized"] = len(output)
        print(f"Wrote {family}: {len(output):,} rows")

    manifest = {
        "version": 1, "purpose": "frozen text-space robustness evaluation",
        "source_file": str(args.source_file), "source_sha256": source_sha256(args.source_file),
        "base_rows": len(frame), "families": families, "severities": SEVERITIES,
        "replicates": args.replicates, "seed": args.seed, "emit_canonicalized": args.emit_canonicalized,
        "canonicalization": "NFKC + format-character removal + Cyrillic homoglyph reversal + conservative separator and leetspeak repair",
        "rows_by_file": counts,
        "label_preservation": "All transformations are character-level and retain the source label.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
