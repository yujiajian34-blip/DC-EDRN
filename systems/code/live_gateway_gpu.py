"""Live-clock, single-GPU request queue and dynamic-batching experiment.

The producer releases requests at real wall-clock deadlines. A separate worker
actually queues, batches, and executes the existing optimized DC-EDRN model.
This is a local screening worker, not an HTTP, multi-GPU, or distributed system.
No training, checkpoint selection, or threshold calibration occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import threading
import time
import traceback
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import ROOT, encode_texts, environment, sha
from measure import frame, load_model, parameters


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values, p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def arrival_offsets(rate: float, duration: float, pattern: str, seed: int) -> list[float]:
    """Construct a trace before either policy runs, so paired runs share arrivals."""
    if pattern == "periodic":
        return (np.arange(int(round(rate * duration))) / rate).tolist()
    if pattern != "bursty":
        raise ValueError(pattern)
    rng = np.random.default_rng(seed)
    offsets: list[float] = []
    whole_seconds = math.ceil(duration)
    for second in range(whole_seconds):
        width = min(1.0, duration - second)
        if width <= 0:
            break
        count = int(rng.poisson(rate * width))
        # A synthetic ON/OFF-like burst: 80% of arrivals occur in the first
        # 20% of each second, with the remainder spread over the rest.
        early = int(rng.binomial(count, 0.8))
        times = np.concatenate((
            rng.uniform(0.0, 0.2 * width, early),
            rng.uniform(0.2 * width, width, count - early),
        ))
        offsets.extend((second + times).tolist())
    return sorted(offsets)


@dataclass
class Request:
    request_id: int
    sample_index: int
    target_time: float
    arrival_time: float


def run_live(
    model,
    texts: list[str],
    device: torch.device,
    offsets: list[float],
    policy: str,
    max_batch: int,
    max_wait_ms: float,
    duration_s: float,
    seed: int,
) -> tuple[dict, list[dict], list[dict]]:
    pending: deque[Request] = deque()
    condition = threading.Condition()
    producer_done = False
    abort = threading.Event()
    errors: list[str] = []
    request_rows: list[dict] = []
    batch_rows: list[dict] = []
    order = np.random.default_rng(seed).permutation(len(texts)).tolist()
    start_at = time.perf_counter() + 0.10

    def producer() -> None:
        nonlocal producer_done
        try:
            for index, offset in enumerate(offsets):
                target = start_at + offset
                remaining = target - time.perf_counter()
                # Event.wait(timeout) on the measured Windows host rounded
                # sub-16-ms waits up to the next scheduler tick. sleep() has
                # substantially finer timing there; abort is checked after it.
                if remaining > 0:
                    time.sleep(remaining)
                if abort.is_set():
                    break
                arrived = time.perf_counter()
                with condition:
                    pending.append(Request(index, order[index % len(order)], target, arrived))
                    condition.notify()
        except BaseException:
            errors.append("PRODUCER:\n" + traceback.format_exc())
            abort.set()
        finally:
            with condition:
                producer_done = True
                condition.notify_all()

    def worker() -> None:
        try:
            while True:
                with condition:
                    condition.wait_for(lambda: bool(pending) or producer_done or abort.is_set())
                    if abort.is_set():
                        return
                    if not pending and producer_done:
                        return
                    first = pending[0]
                    deadline = first.arrival_time + max_wait_ms / 1000.0
                    ready_before_deadline = time.perf_counter() < deadline

                # Avoid Condition.wait(timeout): on Windows its 5-ms timeout
                # was observed to become roughly 16 ms. Short sleep polling
                # releases the queue lock so producer arrivals remain live.
                while True:
                    with condition:
                        if abort.is_set():
                            return
                        remaining = deadline - time.perf_counter()
                        if len(pending) >= max_batch or remaining <= 0:
                            batch = [pending.popleft() for _ in range(min(max_batch, len(pending)))]
                            break
                    time.sleep(min(0.001, remaining))

                # The queue lock is released during CPU encoding and GPU work,
                # so new requests can arrive while the worker is busy.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                cpu_inputs = encode_texts([texts[r.sample_index] for r in batch])
                gpu_inputs = tuple(x.to(device) for x in cpu_inputs)
                with torch.inference_mode():
                    scores = torch.sigmoid(model(*gpu_inputs)["logits"]).float().cpu().numpy()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                finished = time.perf_counter()
                if len(scores) != len(batch) or not np.isfinite(scores).all():
                    raise RuntimeError("Non-finite or incomplete model scores")

                batch_rows.append({
                    "batch_index": len(batch_rows),
                    "batch_size": len(batch),
                    "service_ms": (finished - started) * 1000.0,
                    "dispatch_overshoot_ms": (
                        max(0.0, (started - deadline) * 1000.0)
                        if ready_before_deadline else None
                    ),
                    "started_offset_s": started - start_at,
                    "finished_offset_s": finished - start_at,
                })
                for req in batch:
                    request_rows.append({
                        "request_id": req.request_id,
                        "sample_index": req.sample_index,
                        "target_offset_s": req.target_time - start_at,
                        "arrival_offset_s": req.arrival_time - start_at,
                        "arrival_lag_ms": (req.arrival_time - req.target_time) * 1000.0,
                        "queue_wait_ms": (started - req.arrival_time) * 1000.0,
                        "e2e_ms": (finished - req.arrival_time) * 1000.0,
                        "batch_index": len(batch_rows) - 1,
                    })
        except BaseException:
            errors.append("WORKER:\n" + traceback.format_exc())
            abort.set()
            with condition:
                condition.notify_all()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    worker_thread = threading.Thread(target=worker, name="screening-worker")
    producer_thread = threading.Thread(target=producer, name="request-producer")
    worker_thread.start()
    producer_thread.start()
    producer_thread.join()
    worker_thread.join()
    if errors:
        raise RuntimeError("\n".join(errors))
    if len(request_rows) != len(offsets) or not request_rows:
        raise RuntimeError(f"Incomplete run: {len(request_rows)} / {len(offsets)} requests")

    e2e = [r["e2e_ms"] for r in request_rows]
    queue_wait = [r["queue_wait_ms"] for r in request_rows]
    arrival_lag = [r["arrival_lag_ms"] for r in request_rows]
    dispatch_overshoot = [b["dispatch_overshoot_ms"] for b in batch_rows
                          if b["dispatch_overshoot_ms"] is not None]
    last_finish = max(b["finished_offset_s"] for b in batch_rows)
    elapsed = max(duration_s, last_finish)
    summary = {
        "policy": policy,
        "requests": len(request_rows),
        "offered_duration_s": duration_s,
        "actual_offered_rate_texts_s": len(request_rows) / duration_s,
        "achieved_throughput_texts_s": len(request_rows) / elapsed,
        "drain_after_arrivals_s": max(0.0, last_finish - duration_s),
        "max_batch": max_batch,
        "max_wait_ms": max_wait_ms,
        "mean_batch": float(np.mean([b["batch_size"] for b in batch_rows])),
        "max_observed_batch": max(b["batch_size"] for b in batch_rows),
        "mean_service_ms": float(np.mean([b["service_ms"] for b in batch_rows])),
        "p50_queue_ms": percentile(queue_wait, 50),
        "p95_queue_ms": percentile(queue_wait, 95),
        "p50_e2e_ms": percentile(e2e, 50),
        "p95_e2e_ms": percentile(e2e, 95),
        "p95_arrival_lag_ms": percentile(arrival_lag, 95),
        "p95_dispatch_overshoot_ms": (
            percentile(dispatch_overshoot, 95) if dispatch_overshoot else None
        ),
        "peak_gpu_allocated_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1048576) if device.type == "cuda" else None
        ),
        "batches": len(batch_rows),
    }
    return summary, request_rows, batch_rows


def save_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rates", type=float, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--patterns", choices=["periodic", "bursty"], nargs="+",
                        default=["periodic", "bursty"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.duration <= 0 or any(rate <= 0 for rate in args.rates):
        parser.error("duration and rates must be positive")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be >= 1 and warmup must be >= 0")

    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select the GPU-enabled PyTorch environment")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    # Keep the first run intact: its Windows condition waits were coarsened
    # to ~16 ms, so it must not be silently replaced by the corrected run.
    output = ROOT / ("verification/live_gateway_timerfix_smoke" if device.type == "cpu"
                     else "results/live_gateway_timerfix")
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        parity_path = ROOT / "results/performance/parity.json"
        if not parity_path.exists():
            raise RuntimeError("Missing prior GPU parity result. Run the performance supplement first.")
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if parity.get("status") != "PASS" or parity.get("chosen_optimization") != "precompute":
            raise RuntimeError("A passing GPU precompute parity result is required")
    frame_test = frame()
    frame_test = frame_test[frame_test["split"] == "diagnostic_test"]
    sample = frame_test.groupby("label", group_keys=False).sample(n=64, random_state=20260917)
    sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)
    save_csv(output / "sample_ids.csv", sample[["id", "label", "split"]].to_dict("records"))
    texts = sample["text"].astype(str).tolist()
    configs = json.loads((ROOT / "configs.json").read_text(encoding="utf-8-sig"))
    model, training_parameters = load_model("DC_EDRN_optimized", configs, device, optimization="precompute")
    model.eval()
    for batch_size in (1, 8, 32):
        batch = [texts[i % len(texts)] for i in range(batch_size)]
        for _ in range(args.warmup):
            cpu_inputs = encode_texts(batch)
            gpu_inputs = tuple(x.to(device) for x in cpu_inputs)
            with torch.inference_mode():
                _ = torch.sigmoid(model(*gpu_inputs)["logits"]).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    manifest = {
        "status": "RUNNING",
        "generated_at_utc": utc_now(),
        "device": str(device),
        "environment": environment(device),
        "platform_machine": platform.machine(),
        "model": "DC_EDRN_optimized",
        "inference_parameters": parameters(model),
        "training_parameters": training_parameters,
        "training_performed": False,
        "thresholds_changed": False,
        "checkpoint_sha256": sha(ROOT / "reference_checkpoints/DC_EDRN_seed42.pth"),
        "script_sha256": sha(Path(__file__)),
        "sample_sha256": sha(output / "sample_ids.csv"),
        "duration_s": args.duration,
        "rates_texts_s": args.rates,
        "patterns": args.patterns,
        "repeats": args.repeats,
        "warmup_per_batch": args.warmup,
        "cpu_threads": 4,
        "method": "Live-clock producer thread and separate single-GPU worker thread. Synthetic periodic and bursty arrivals; local encoding, host-to-device transfer, inference and score return included. Requests are drained after the offered interval. Windows arrival deadlines use time.sleep; batching deadlines use <=1-ms sleep polling outside the queue lock.",
        "scope": "In-process local screening worker; no HTTP, network, downstream LLM, multi-GPU, or distributed deployment.",
    }
    summary_rows: list[dict] = []
    request_rows: list[dict] = []
    batch_rows: list[dict] = []
    policies = [("immediate", 1, 0.0), ("dynamic32_5ms", 32, 5.0)]
    try:
        for pattern in args.patterns:
            for rate in args.rates:
                for repeat in range(args.repeats):
                    seed = 20260919 + repeat
                    offsets = arrival_offsets(rate, args.duration, pattern, seed)
                    if not offsets:
                        raise RuntimeError("Arrival trace contains no requests")
                    # Alternate order to reduce systematic first/second-run bias.
                    order = policies if repeat % 2 == 0 else list(reversed(policies))
                    for policy, max_batch, wait_ms in order:
                        print(f"LIVE {pattern} {rate:g}/s seed={seed} {policy}: {len(offsets)} requests", flush=True)
                        result, requests, batches = run_live(
                            model, texts, device, offsets, policy, max_batch, wait_ms,
                            args.duration, seed,
                        )
                        tag = {"pattern": pattern, "target_rate_texts_s": rate,
                               "repeat": repeat, "seed": seed}
                        summary_rows.append({**tag, **result})
                        request_rows.extend([{**tag, "policy": policy, **row} for row in requests])
                        batch_rows.extend([{**tag, "policy": policy, **row} for row in batches])
                        save_csv(output / "summary.csv", summary_rows)
                        print(f"  achieved={result['achieved_throughput_texts_s']:.1f}/s; "
                              f"P95={result['p95_e2e_ms']:.1f}ms; "
                              f"arrival-lag-P95={result['p95_arrival_lag_ms']:.1f}ms", flush=True)
        save_csv(output / "requests.csv", request_rows)
        save_csv(output / "batches.csv", batch_rows)
        manifest["status"] = "COMPLETE"
        manifest["completed_at_utc"] = utc_now()
        manifest["run_count"] = len(summary_rows)
        manifest["timing_warning"] = any(
            row["p95_arrival_lag_ms"] > 5.0 or
            (row["p95_dispatch_overshoot_ms"] is not None and
             row["p95_dispatch_overshoot_ms"] > 5.0)
            for row in summary_rows
        )
    except BaseException:
        manifest["status"] = "FAILED"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        archive = output / "LIVE_GATEWAY_RESULTS_TO_RETURN.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for file in sorted(output.glob("*.csv")):
                z.write(file, file.name)
            z.write(output / "manifest.json", "manifest.json")
            z.write(Path(__file__), "code/live_gateway_gpu.py")
        print(f"RETURN: {archive}", flush=True)


if __name__ == "__main__":
    main()
