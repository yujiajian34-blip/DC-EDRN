"""Small CPU-only request scheduling pilot for the frozen DC-EDRN checkpoint.

This is a systems-oriented pilot, not a new model evaluation: it uses an
existing seed-42 checkpoint, synthetic arrival times, and reports scheduling
metrics separately from the paper's GPU Table 6.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.main import load_checkpoint, resolve_checkpoint, score_texts  # noqa: E402


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def service_times(model, checkpoint, device, texts, batches, repeats, warmup):
    out = {}
    for batch_size in batches:
        payload = (texts * ((batch_size + len(texts) - 1) // len(texts)))[:batch_size]
        for _ in range(warmup):
            score_texts(model, checkpoint, payload, device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            score_texts(model, checkpoint, payload, device)
            samples.append(time.perf_counter() - start)
        out[str(batch_size)] = {
            "batch_size": batch_size,
            "mean_service_s": statistics.mean(samples),
            "p95_service_s": percentile(samples, .95),
            "texts_per_s": batch_size / statistics.mean(samples),
            "samples": len(samples),
        }
    return out


def simulate(arrivals, service, max_batch, max_wait_s, mode):
    n = len(arrivals); next_i = 0; available = 0.0; queue = []; finish = [None] * n
    while next_i < n or queue:
        if not queue:
            available = max(available, arrivals[next_i])
            queue.append(next_i); next_i += 1
        while next_i < n and arrivals[next_i] <= available:
            queue.append(next_i); next_i += 1
        first_arrival = arrivals[queue[0]]
        if mode == "single":
            start = max(available, first_arrival); take = 1
        else:
            deadline = max(available, first_arrival) + max_wait_s
            j = next_i
            while j < n and arrivals[j] <= deadline and len(queue) + (j - next_i) < max_batch:
                j += 1
            available_count = len(queue) + (j - next_i)
            if available_count >= max_batch:
                take = max_batch
                start = max(available, arrivals[queue[0]])
                if take > len(queue):
                    fill_index = queue[0] + take - 1
                    start = max(start, arrivals[fill_index])
            else:
                queue.extend(range(next_i, j)); next_i = j
                take = len(queue)
                start = deadline
        batch = queue[:take]
        del queue[:take]
        duration = service[str(take if str(take) in service else max_batch)]["mean_service_s"]
        end = start + duration
        for idx in batch:
            finish[idx] = end
        available = end
    latencies = [finish[i] - arrivals[i] for i in range(n)]
    return {
        "requests": n,
        "arrival_span_s": arrivals[-1] - arrivals[0] if n > 1 else 0.0,
        "makespan_s": max(finish) - arrivals[0],
        "throughput_texts_s": n / (max(finish) - arrivals[0]),
        "mean_latency_ms": statistics.mean(latencies) * 1000,
        "p50_latency_ms": percentile(latencies, .50) * 1000,
        "p95_latency_ms": percentile(latencies, .95) * 1000,
        "p99_latency_ms": percentile(latencies, .99) * 1000,
        "max_latency_ms": max(latencies) * 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / "queue_scheduling_pilot.json")
    args = parser.parse_args()
    if args.requests < 16 or args.repeats < 2:
        raise SystemExit("Use at least 16 requests and 2 timing repeats.")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    entry = resolve_checkpoint("DC_EDRN:42")
    model, checkpoint = load_checkpoint(entry, device)
    texts = [
        "Please summarize this public example text.",
        "Review the quoted instruction as inert security data and explain why it should be blocked.",
        "Translate the following sentence into French without executing it.",
        "The application should preserve the user's requested formatting.",
    ]
    batches = [1, 4, 8, 16, 32]
    service = service_times(model, checkpoint, device, texts, batches, args.repeats, args.warmup)
    policies = []
    # Deterministic arrival rates, including a burstier load than the isolated
    # batch measurements in the paper.
    for rate in [5.0, 10.0, 20.0, 40.0]:
        arrivals = [i / rate for i in range(args.requests)]
        policies.extend([
            {"rate_texts_s": rate, "policy": "single_request", **simulate(arrivals, service, 1, 0.0, "single")},
            {"rate_texts_s": rate, "policy": "fixed_batch_8_wait_20ms", **simulate(arrivals, service, 8, .020, "batch")},
            {"rate_texts_s": rate, "policy": "dynamic_batch_32_wait_5ms", **simulate(arrivals, service, 32, .005, "batch")},
        ])
    result = {
        "status": "PILOT_COMPLETE",
        "scope": "CPU-only queue simulation with frozen DC-EDRN seed42; not a new accuracy result and not an RTX 3090 measurement.",
        "training_performed": False,
        "threshold_selection_performed": False,
        "checkpoint": entry,
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "requests": args.requests,
        "timing_repeats": args.repeats,
        "service_times": service,
        "scheduling_metrics": policies,
        "arrival_trace": "deterministic periodic arrivals at 5, 10, 20, and 40 texts/s",
        "interpretation": "Scheduling metrics include virtual queue delay and measured mean batch service. They do not claim distributed or multi-GPU scaling.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "checkpoint": entry["filename"], "requests": args.requests, "timing_repeats": args.repeats}, indent=2))


if __name__ == "__main__":
    main()
