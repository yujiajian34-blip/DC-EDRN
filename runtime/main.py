"""Portable FP32 inference from unchanged historical checkpoints.

Run from the public release root: python -m runtime.main --text "Example text"
This entry point never trains a model or selects/refits a threshold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Iterator

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.code import common

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = "DC_EDRN:42"
PREFLIGHT_TEXT = "Please summarize this public example text."


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_entries() -> list[dict]:
    manifest = json.loads((ROOT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    return manifest["checkpoints"]


def resolve_checkpoint(selector: str) -> dict:
    for entry in checkpoint_entries():
        if selector in (entry["selector"], entry["filename"], Path(entry["filename"]).stem):
            return entry
    raise ValueError(f"Unknown bundled checkpoint {selector!r}; use --list-checkpoints.")


def load_checkpoint(entry: dict, device: torch.device):
    """Use only the bundled, checksum-verified weights and their own config."""
    path = ROOT / "checkpoints" / entry["filename"]
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {entry['filename']}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"config", "state_dict", "downsample", "max_bytes", "threshold", "seed"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Incomplete checkpoint: {entry['filename']}")
    threshold = float(checkpoint["threshold"])
    if not math.isfinite(threshold):
        raise ValueError("Stored checkpoint threshold is not finite.")
    if threshold != entry["threshold"] or int(checkpoint["seed"]) != entry["seed"]:
        raise ValueError("Checkpoint metadata does not match its release manifest.")
    if checkpoint["config"]["name"] != entry["original_config_name"]:
        raise ValueError("Checkpoint configuration name does not match its release manifest.")
    # Retain every original module, including the auxiliary duration head.
    model = common.MATCHED.MatchedLocalEncoderModel(
        checkpoint["config"], int(checkpoint["downsample"])
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def score_texts(model, checkpoint: dict, texts: list[str], device: torch.device) -> list[float]:
    # The historical primary protocol used 512 UTF-8 bytes and 128 lexical tokens.
    # max_bytes and word_vocab come from the saved checkpoint, not a new config.
    byte_ids, word_ids = common.encode_texts(
        texts,
        max_bytes=int(checkpoint["max_bytes"]),
        max_words=128,
        vocab=int(checkpoint["config"]["word_vocab"]),
    )
    outputs = model(byte_ids.to(device), word_ids.to(device))
    scores = torch.sigmoid(outputs["logits"]).float().cpu().tolist()
    if not all(math.isfinite(score) for score in scores):
        raise FloatingPointError("Nonfinite model score; no prediction was emitted.")
    return scores


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on input line {line_number}: {error.msg}") from error
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Input line {line_number} requires a JSON object with a string 'text'.")
            yield {"id": record.get("id", line_number), "text": record["text"]}


def prediction_rows(records, model, checkpoint: dict, entry: dict, device, batch_size: int):
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield from predict_batch(batch, model, checkpoint, entry, device)
            batch = []
    if batch:
        yield from predict_batch(batch, model, checkpoint, entry, device)


def predict_batch(batch, model, checkpoint, entry, device):
    texts = [record["text"] for record in batch]
    scores = score_texts(model, checkpoint, texts, device)
    threshold = float(checkpoint["threshold"])
    for record, score in zip(batch, scores):
        text = record["text"]
        yield {
            "id": record["id"],
            "model": entry["model"],
            "original_config_name": checkpoint["config"]["name"],
            "checkpoint": entry["filename"],
            "seed": int(checkpoint["seed"]),
            "score": score,
            "threshold": threshold,
            "prediction": int(score >= threshold),
            "byte_truncated": len(text.encode("utf-8")) > int(checkpoint["max_bytes"]),
            "word_truncated": len(common.V7.WORD_PATTERN.findall(text.casefold())) > 128,
        }


def preflight() -> dict:
    """One synthetic text per checkpoint; this is not a benchmark evaluation."""
    entries = checkpoint_entries()
    if len(entries) != 18 or len({entry["selector"] for entry in entries}) != 18:
        raise ValueError("Expected exactly 18 distinct primary checkpoint selectors.")
    device = torch.device("cpu")
    results = []
    for entry in entries:
        model, checkpoint = load_checkpoint(entry, device)
        score = score_texts(model, checkpoint, [PREFLIGHT_TEXT], device)[0]
        parameter_count = common.parameters(model)
        if parameter_count != entry["parameter_count"]:
            raise ValueError(f"Parameter-count mismatch: {entry['selector']}")
        results.append({
            "selector": entry["selector"],
            "checkpoint": entry["filename"],
            "sha256": entry["sha256"],
            "strict_state_dict_load": True,
            "parameter_count": parameter_count,
            "threshold": float(checkpoint["threshold"]),
            "dummy_score": score,
            "finite_score": True,
        })
        del model
    return {
        "status": "passed",
        "scope": "CPU strict load plus one synthetic-text forward pass for each of 18 original checkpoints",
        "training_performed": False,
        "threshold_selection_performed": False,
        "benchmark_evaluation_performed": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "numpy", "pandas", "scikit-learn", "tqdm"
        )},
        "synthetic_text": PREFLIGHT_TEXT,
        "checkpoint_count": len(results),
        "checks": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="A single text to score, including an empty string.")
    source.add_argument("--input", type=Path, help="UTF-8 JSONL with one {'text': str, 'id': optional} per line.")
    source.add_argument("--list-checkpoints", action="store_true", help="Print all bundled selectors and metadata.")
    source.add_argument("--preflight", action="store_true", help="Check all 18 checkpoints on one dummy text, CPU only.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Model:seed selector, checkpoint filename, or stem.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Write JSONL predictions or JSON preflight/list output to this file.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("--batch-size and --cpu-threads must be positive integers")
    if args.preflight and args.device != "cpu":
        parser.error("--preflight is deliberately limited to CPU")
    if args.input and args.output and args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    if args.output:
        protected = (ROOT / "checkpoints").resolve()
        if args.output.resolve().is_relative_to(protected):
            parser.error("--output cannot overwrite bundled checkpoint files")
    return args


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(args.cpu_threads)
    if args.list_checkpoints:
        result = json.dumps(checkpoint_entries(), indent=2, ensure_ascii=False)
    elif args.preflight:
        result = json.dumps(preflight(), indent=2, ensure_ascii=False)
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
        if args.device == "cuda":
            # Same attention-backend policy as the historical matched-model entry point.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        device = torch.device(args.device)
        entry = resolve_checkpoint(args.checkpoint)
        model, checkpoint = load_checkpoint(entry, device)
        records = read_jsonl(args.input) if args.input else [{"id": 1, "text": args.text}]
        rows = prediction_rows(records, model, checkpoint, entry, device, args.batch_size)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            for row in rows:
                print(json.dumps(row, ensure_ascii=False, allow_nan=False))
        return
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
