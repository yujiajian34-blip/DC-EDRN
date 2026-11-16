"""No-training parity and performance supplement for the existing seed-42 models."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import ROOT, dump, encode_texts, environment, inference_copy, model, parameters, sha
from optimized_dc import optimized_copy

NAMES = ("DC_EDRN", "BiLSTM", "ByteCNN")
VARIANTS = ("DC_EDRN_original", "DC_EDRN_optimized", "BiLSTM", "ByteCNN")
ATOL, RTOL = 1e-5, 1e-5


def now():
    return datetime.now(timezone.utc).isoformat()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clean(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def checkpoint_path(name):
    return ROOT / "reference_checkpoints" / f"{name}_seed42.pth"


def checkpoint(name):
    # These bundled checkpoints are trusted, user-supplied experiment artifacts.
    c = torch.load(checkpoint_path(name), map_location="cpu", weights_only=True)
    if "state_dict" not in c or "threshold" not in c:
        raise ValueError(f"Checkpoint is missing state_dict/threshold: {name}")
    return c


def load_model(variant, configs, device, optimization="precompute"):
    name = "DC_EDRN" if variant.startswith("DC_EDRN_") else variant
    c = checkpoint(name)
    original = model(configs[name]).float().eval()
    original.load_state_dict(c["state_dict"], strict=True)
    training_count = parameters(original)
    m = inference_copy(original)
    if variant == "DC_EDRN_optimized":
        m = optimized_copy(m, optimization=optimization)
    del original, c
    m = m.to(device).float().eval()
    for layer in m.modules():
        if isinstance(layer, torch.nn.RNNBase):
            layer.flatten_parameters()
    return m, training_count


def thresholds(name):
    result = {"original_threshold": float(checkpoint(name)["threshold"])}
    path = ROOT / "reference_thresholds.json"
    if path.exists():
        table = json.loads(path.read_text(encoding="utf-8-sig"))
        entry = table.get(name, {})
        if not isinstance(entry, dict):
            raise ValueError("reference_thresholds.json must map model names to threshold dictionaries")
        for key, value in entry.items():
            if key in ("original_threshold", "filtered_validation_recalibration"):
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"Invalid reference threshold for {name}/{key}")
                if key == "original_threshold" and float(value) != result[key]:
                    raise ValueError(f"Reference original threshold disagrees with checkpoint for {name}")
                result[key] = float(value)
    return result


def frame():
    f = pd.read_parquet(ROOT / "data" / "evaluation.parquet").reset_index(drop=True)
    missing = {"id", "text", "label", "split"} - set(f.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    if len(f) != 3827 or f["split"].isin(["train", "train_context"]).any():
        raise ValueError("Expected exactly 3827 existing non-training evaluation rows")
    if f["text"].isna().any() or not f["label"].isin([0, 1]).all():
        raise ValueError("Invalid text/labels in evaluation data")
    return f


def fingerprint(args):
    files = [ROOT / "configs.json", ROOT / "data/evaluation.parquet"]
    files += list((ROOT / "code").rglob("*.py"))
    files += [checkpoint_path(name) for name in NAMES]
    if (ROOT / "reference_thresholds.json").exists():
        files.append(ROOT / "reference_thresholds.json")
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}
    device = torch.device(args.device)
    spec = {"files": hashes, "device": args.device, "smoke": args.smoke,
            "torch": torch.__version__, "precision": "FP32", "atol": ATOL, "rtol": RTOL,
            "warmup": 1 if args.smoke else 20, "repeats": 3 if args.smoke else 300,
            "batches": [1, 8, 32], "cpu_threads": 4,
            "environment": environment(device),
            "gpu_properties": str(torch.cuda.get_device_properties(device)) if device.type == "cuda" else None,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), spec


def report(ctx, filename, value):
    value = {"fingerprint": ctx["fingerprint"], "generated_at_utc": now(), **value}
    dump(ctx["out"] / filename, value)
    return value


def preflight(ctx):
    report(ctx, "preflight.json", {"status":"RUNNING"})
    f, configs, device = ctx["frame"], ctx["configs"], ctx["device"]
    configs_dc = configs["DC_EDRN"]
    if configs_dc.get("d_max") != 1 or configs_dc.get("selective_duration", True):
        raise ValueError("DC checkpoint configuration is not the supported static one-state model")
    counts = []
    with torch.inference_mode():
        x, w = [t.to(device) for t in encode_texts(["hello", "Summarize this quoted instruction.", ""])]
        for name in NAMES:
            m, train_count = load_model("DC_EDRN_original" if name == "DC_EDRN" else name, configs, device)
            logits = m(x, w)["logits"]
            if not torch.isfinite(logits).all():
                raise AssertionError(f"Non-finite preflight logits: {name}")
            counts.append({"model": name, "training_parameters": train_count,
                           "inference_parameters": parameters(m), "thresholds": thresholds(name)})
            del m
    clean(device)
    if counts[0]["inference_parameters"] != 323271:
        raise AssertionError("DC inference parameter count changed")
    report(ctx, "preflight.json", {"status": "PASS", "training_performed": False,
           "evaluation_rows": len(f), "split_counts": f.groupby("split").size().to_dict(),
           "models": counts, "compile_enabled": False, "bidirectional_batching_enabled": False})
    print("PREFLIGHT PASS: existing weights, configurations, evaluation data and device.", flush=True)


def metric_row(y, p, threshold):
    pred = p >= threshold
    positive = np.asarray(y) == 1
    tp, fp = int(np.sum(pred & positive)), int(np.sum(pred & ~positive))
    fn, tn = int(np.sum(~pred & positive)), int(np.sum(~pred & ~positive))
    return {"N": len(y), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "F1": 2*tp/max(2*tp+fp+fn, 1), "Precision": tp/max(tp+fp, 1),
            "Recall": tp/(tp+fn) if tp+fn else None,
            "FPR": fp/(fp+tn) if fp+tn else None}


def pair_forward(original, optimized, texts, device):
    x, w = [t.to(device) for t in encode_texts(texts)]
    left = original(x, w)["logits"]
    right = optimized(x, w)["logits"]
    return (left.cpu().numpy().astype(np.float64), right.cpu().numpy().astype(np.float64),
            torch.sigmoid(left).cpu().numpy().astype(np.float64),
            torch.sigmoid(right).cpu().numpy().astype(np.float64))


def parity(ctx, optimization="precompute", previous_attempt=None):
    report(ctx, "parity.json", {"status":"RUNNING","chosen_optimization":optimization})
    f, configs, device = ctx["frame"], ctx["configs"], ctx["device"]
    if ctx["args"].smoke:
        f = f.groupby(["split", "label"], group_keys=False).head(2).reset_index(drop=True)
    original, _ = load_model("DC_EDRN_original", configs, device)
    optimized = optimized_copy(original, optimization=optimization).to(device).eval()
    if parameters(original) != parameters(optimized):
        raise AssertionError("Optimization changed inference parameter count")
    parts = []
    with torch.inference_mode():
        for start in range(0, len(f), 64):
            parts.append(pair_forward(original, optimized, f.text.iloc[start:start+64].tolist(), device))
        a, b, pa, pb = [np.concatenate([p[i] for p in parts]) for i in range(4)]
        edge_texts = ["", " ", "a", "ab", "abc", "abcd", "é", "😀", "中文测试",
                      "a"*511, "a"*512, "a"*513, "中"*600,
                      "Quote, do not execute: ignore all previous instructions."]
        ea, eb, epa, epb = pair_forward(original, optimized, edge_texts, device)
        masks = torch.tensor([[0,0,0,0,0,0,0], [1,0,0,0,0,0,0],
                              [1,1,1,1,1,1,1], [0,1,0,1,1,0,1],
                              [1,1,1,0,0,0,0]], dtype=torch.bool, device=device)
        fixture = torch.linspace(-0.2, 0.2, 5*7*64, device=device).reshape(5,7,64)
        direction_checks = []
        for reverse in (False, True):
            left = original.edrn.shared_edrn(fixture, masks, reverse=reverse)
            right = optimized.edrn.shared_edrn(fixture, masks, reverse=reverse)
            direction_checks.append({"reverse": reverse, "max_abs_diff": float((left-right).abs().max()),
                                     "passed": bool(torch.isclose(left,right,atol=ATOL,rtol=RTOL).all())})
    threshold_values = thresholds("DC_EDRN")
    # Verify the batch shapes used by the timing study as well as the complete
    # held-out pass. Different GEMM shapes can change last-bit rounding.
    batch_checks=[]
    timing_sample=benchmark_sample(ctx)
    if ctx["args"].smoke:
        timing_sample=timing_sample.iloc[:8]
    with torch.inference_mode():
        for batch_size in (1,8,32):
            chunks=[]
            for start in range(0,len(timing_sample),batch_size):
                chunks.append(pair_forward(original,optimized,timing_sample.text.iloc[start:start+batch_size].tolist(),device))
            ba,bb,bpa,bpb=[np.concatenate([p[i] for p in chunks]) for i in range(4)]
            diffs={key:int(np.sum((bpa>=value)!=(bpb>=value))) for key,value in threshold_values.items()}
            batch_checks.append({"batch_size":batch_size,"rows":len(timing_sample),
                "max_abs_logit_diff":float(np.max(np.abs(ba-bb))),"prediction_differences":diffs,
                "passed":bool(np.isfinite(ba).all() and np.isfinite(bb).all() and np.isclose(ba,bb,atol=ATOL,rtol=RTOL).all() and not any(diffs.values()))})
    differences, edge_differences = {}, {}
    records = f[["id","split","label"]].copy()
    records["original_logit"], records["optimized_logit"] = a, b
    records["original_score"], records["optimized_score"] = pa, pb
    metrics = []
    for protocol, threshold in threshold_values.items():
        differences[protocol] = int(np.sum((pa >= threshold) != (pb >= threshold)))
        edge_differences[protocol] = int(np.sum((epa >= threshold) != (epb >= threshold)))
        records[protocol+"_original_prediction"] = (pa >= threshold).astype(int)
        records[protocol+"_optimized_prediction"] = (pb >= threshold).astype(int)
        for split, group in f.groupby("split"):
            indices = group.index.to_numpy()
            for variant, scores in (("DC_EDRN_original",pa), ("DC_EDRN_optimized",pb)):
                metrics.append({"Model": variant, "Split": split, "ThresholdProtocol": protocol,
                                "Threshold": threshold, **metric_row(group.label, scores[indices], threshold)})
    finite = bool(all(np.isfinite(x).all() for x in (a,b,pa,pb,ea,eb,epa,epb)))
    close = bool(np.isclose(a,b,atol=ATOL,rtol=RTOL).all())
    edge_close = bool(np.isclose(ea,eb,atol=ATOL,rtol=RTOL).all())
    passed = finite and close and edge_close and not any(differences.values()) and not any(edge_differences.values()) and all(x["passed"] for x in direction_checks) and all(x["passed"] for x in batch_checks)
    records.to_parquet(ctx["out"] / "paired_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_csv(ctx["out"] / "parity_metrics.csv", index=False)
    result = report(ctx, "parity.json", {"status": "PASS" if passed else "FAIL",
        "chosen_optimization": optimization, "previous_attempt": previous_attempt,
        "smoke_only": ctx["args"].smoke, "evaluated_rows": len(f), "expected_formal_rows":3827,
        "batch_size":64, "logit_atol":ATOL, "logit_rtol":RTOL, "finite":finite,
        "max_abs_logit_diff":float(np.max(np.abs(a-b))),
        "max_abs_score_diff":float(np.max(np.abs(pa-pb))), "logits_close":close,
        "thresholds":threshold_values, "prediction_differences":differences,
        "edge_cases": {"count":len(edge_texts),"max_abs_logit_diff":float(np.max(np.abs(ea-eb))),
                       "logits_close":edge_close,"prediction_differences":edge_differences},
        "mask_direction_cases":direction_checks,"timing_batch_cases":batch_checks, "parameters_identical":True,
        "architecture_unchanged":True,"training_performed":False,
        "scope":"Algebraic implementation consistency; not retraining or a statistical convergence proof."})
    del original, optimized
    clean(device)
    print(f"PARITY {result['status']}: {len(f)} rows; max |logit diff|={result['max_abs_logit_diff']:.3g}; prediction differences={differences}",flush=True)
    if not passed:
        if optimization == "precompute":
            report(ctx, "parity_precompute_attempt.json", result)
            print("Precomputation parity failed; trying conservative invariant hoisting with original per-step arithmetic. Thresholds and tolerances remain unchanged.",flush=True)
            return parity(ctx, optimization="static_hoist", previous_attempt={
                "optimization":"precompute","status":"FAIL","report":"parity_precompute_attempt.json"})
        raise RuntimeError("Parity failed. Performance/profile stages are blocked; no thresholds or weights were changed.")


def benchmark_sample(ctx):
    f = ctx["frame"]
    f = f[f.split == "diagnostic_test"]
    sample = f.groupby("label", group_keys=False).sample(n=64, random_state=20260917)
    sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)
    sample[["id","label","split"]].to_csv(ctx["out"] / "benchmark_sample_ids.csv", index=False)
    return sample


def require_parity(ctx):
    p = json.loads((ctx["out"] / "parity.json").read_text(encoding="utf-8"))
    if p.get("fingerprint") != ctx["fingerprint"] or p.get("status") != "PASS":
        raise RuntimeError("A current passing parity result is required")
    return p["chosen_optimization"]


def benchmark(ctx):
    optimization = require_parity(ctx)
    report(ctx,"benchmark.json",{"status":"RUNNING","chosen_optimization":optimization})
    device, configs, args = ctx["device"], ctx["configs"], ctx["args"]
    sample = benchmark_sample(ctx)
    warmup, repeats = (1,3) if args.smoke else (20,300)
    rows, raw = [], []
    with torch.inference_mode():
        for variant in VARIANTS:
            print(f"BENCHMARK {variant} ({device}; FP32)",flush=True)
            m, train_count = load_model(variant, configs, device, optimization=optimization)
            for bs in (1,8,32):
                texts = [sample.text.iloc[((np.arange(bs)+i*bs)%len(sample)).tolist()].tolist() for i in range((len(sample)+bs-1)//bs)]
                encoded = [tuple(t.to(device) for t in encode_texts(b)) for b in texts]
                for kind in ("forward_only", "tokenize_transfer_forward_scores"):
                    def once(i):
                        index=i%len(texts)
                        inputs = encoded[index] if kind == "forward_only" else tuple(t.to(device) for t in encode_texts(texts[index]))
                        scores = torch.sigmoid(m(*inputs)["logits"])
                        if kind != "forward_only":
                            scores = scores.cpu().numpy()
                        return scores
                    for i in range(warmup):
                        once(i)
                    sync(device)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    durations=[]
                    for i in range(repeats):
                        sync(device)
                        t=time.perf_counter()
                        once(i)
                        sync(device)
                        elapsed=(time.perf_counter()-t)*1000
                        durations.append(elapsed)
                        raw.append({"Model":variant,"Batch":bs,"Measurement":kind,"Repeat":i,"Elapsed_ms":elapsed})
                    mean=float(np.mean(durations))
                    rows.append({"Model":variant,"Optimization":optimization if variant=="DC_EDRN_optimized" else "original","Device":str(device),"Batch":bs,"Measurement":kind,
                        "Precision":"FP32","Warmup":warmup,"Repeats":repeats,"Mean_ms":mean,
                        "P50_ms":float(np.percentile(durations,50)),"P95_ms":float(np.percentile(durations,95)),
                        "P99_ms":float(np.percentile(durations,99)),"Texts_per_second":bs*1000/mean,
                        "Training_parameters":train_count,"Inference_parameters":parameters(m),
                        "Peak_GPU_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None})
                del encoded
            del m
            clean(device)
    pd.DataFrame(rows).to_csv(ctx["out"] / "screening_cost.csv",index=False)
    pd.DataFrame(raw).to_csv(ctx["out"] / "timing_samples.csv",index=False)
    report(ctx,"benchmark.json",{"status":"PASS","smoke_only":args.smoke,"rows":len(rows),
        "sample_size":len(sample),"warmup":warmup,"repeats":repeats,"batches":[1,8,32],"chosen_optimization":optimization,
        "sample_sha256":sha(ctx["out"] / "benchmark_sample_ids.csv"),
        "device":str(device),"supports_gpu_performance_claims":device.type=="cuda" and not args.smoke,
        "scope":"Local component replay. forward_only uses resident encoded tensors and includes sigmoid; the full component additionally includes text encoding, transfer, and scores copied to CPU. No networking, queueing, or downstream LLM. GPU memory includes the model and resident input buffers. Compilation is disabled."})
    print("BENCHMARK COMPLETE",flush=True)


def profile(ctx):
    optimization = require_parity(ctx)
    report(ctx,"profile_summary.json",{"status":"RUNNING","chosen_optimization":optimization})
    device, configs = ctx["device"], ctx["configs"]
    texts = benchmark_sample(ctx).text.iloc[:1].tolist()
    activity = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activity.append(torch.profiler.ProfilerActivity.CUDA)
    summary=[]
    with torch.inference_mode():
        for variant in VARIANTS[:2]:
            m,_=load_model(variant,configs,device,optimization=optimization)
            inputs=tuple(t.to(device) for t in encode_texts(texts))
            for _ in range(1 if ctx["args"].smoke else 20):
                m(*inputs)
            sync(device)
            with torch.profiler.profile(activities=activity,record_shapes=True,profile_memory=True) as p:
                with torch.profiler.record_function("component_text_encoding_CPU"):
                    cpu_inputs=encode_texts(texts)
                with torch.profiler.record_function("component_host_to_device"):
                    gpu_inputs=tuple(t.to(device) for t in cpu_inputs)
                with torch.profiler.record_function("component_model_forward"):
                    scores=torch.sigmoid(m(*gpu_inputs)["logits"])
                with torch.profiler.record_function("component_scores_to_CPU"):
                    scores=scores.cpu().numpy()
                sync(device)
            # Kineto may silently fail on non-ASCII Windows output paths.
            # Export via an ASCII relative path, validate, then write via pathlib.
            with tempfile.TemporaryDirectory(prefix="dc_edrn_trace_", dir=".") as temp_dir:
                temp_trace = Path(temp_dir) / "trace.json"
                p.export_chrome_trace(str(temp_trace))
                payload = temp_trace.read_bytes()
                if not payload or not isinstance(json.loads(payload), dict):
                    raise RuntimeError("Profiler did not export a valid trace")
                (ctx["out"] / f"profile_{variant}.json").write_bytes(payload)
            events=[]
            for e in p.key_averages():
                events.append({"Name":e.key,"Count":e.count,"Self_CPU_us":e.self_cpu_time_total,
                    "Total_CPU_us":e.cpu_time_total,
                    "Self_device_us":getattr(e,"self_device_time_total",0.0),
                    "Total_device_us":getattr(e,"device_time_total",0.0),
                    "Self_CPU_memory_bytes":e.self_cpu_memory_usage,
                    "Self_device_memory_bytes":getattr(e,"self_device_memory_usage",0)})
            pd.DataFrame(events).to_csv(ctx["out"] / f"profile_{variant}_events.csv",index=False)
            summary.append({"model":variant,"batch":1,"recorded_forward_calls":1,
                            "trace":f"profile_{variant}.json","events":f"profile_{variant}_events.csv"})
            del m,inputs,gpu_inputs,cpu_inputs
            clean(device)
    report(ctx,"profile_summary.json",{"status":"PASS","models":summary,"smoke_only":ctx["args"].smoke,"chosen_optimization":optimization,
        "interpretation":"Diagnostic instrumented traces, not benchmark latency. CPU event time includes dispatch and possible waits; it is not a pure launch-overhead measurement. Inclusive events overlap; GPU events may overlap. Do not add CPU and GPU totals, sum nested inclusive rows, or subtract either total from wall latency to claim an overhead fraction. Use screening_cost.csv for latency."})
    print("PROFILE COMPLETE: interpret traces separately from synchronized latency.",flush=True)


def collect(ctx):
    out=ctx["out"]
    stage_files={"preflight":"preflight.json","parity":"parity.json",
                 "benchmark":"benchmark.json","profile":"profile_summary.json"}
    statuses={}
    for key,filename in stage_files.items():
        path=out/filename
        data=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        statuses[key]=data.get("status","MISSING") if data.get("fingerprint")==ctx["fingerprint"] else "MISSING_OR_STALE"
    complete=all(s=="PASS" for s in statuses.values())
    report(ctx,"completion.json",{"status":"COMPLETE" if complete else "INCOMPLETE","stages":statuses,
        "training_performed":False,"smoke_only":ctx["args"].smoke,
        "supports_gpu_performance_claims":complete and ctx["device"].type=="cuda" and not ctx["args"].smoke})
    archive=out/"PERFORMANCE_RESULTS_TO_RETURN.zip"
    temp=archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.suffix in (".json",".csv",".parquet",".txt"):
                z.write(path,path.name)
        for path in sorted((ROOT/"code").rglob("*.py")):
            z.write(path,path.relative_to(ROOT).as_posix())
        for filename in ("configs.json","reference_thresholds.json","data/data_protocol.json","data/split_counts.csv"):
            path=ROOT/filename
            if path.exists():
                z.write(path,filename)
    temp.replace(archive)
    print(f"RETURN ({'COMPLETE' if complete else 'INCOMPLETE'}): {archive}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["preflight","parity","benchmark","profile","collect","all"],default="all")
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    parser.add_argument("--smoke",action="store_true",help="Tiny workflow check; never use its measurements in the paper")
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the existing CUDA PyTorch interpreter; no packages are installed by this supplement.")
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    out=ROOT/("verification/performance_smoke" if args.smoke else "results/performance")
    out.mkdir(parents=True,exist_ok=True)
    fp,spec=fingerprint(args)
    ctx={"args":args,"device":device,"out":out,"fingerprint":fp,
         "frame":frame(),"configs":json.loads((ROOT/"configs.json").read_text(encoding="utf-8-sig"))}
    run_metadata={"requested_stage":args.stage,"training_performed":False,
        "environment":environment(device),"platform_machine":platform.machine(),"protocol":spec,
        "scope":"Existing seed-42 checkpoint inference only. No fitting, optimization steps, threshold selection, or changes to the original experiment installation.",
        "warning":"SMOKE ONLY" if args.smoke else ("CPU ONLY: not evidence of GPU performance" if device.type=="cpu" else None)}
    if args.stage=="collect":
        # Preserve the original run and failure records, including their exact
        # traceback, when packaging an existing (possibly failed) run.
        collection_metadata={**run_metadata,
            "scope":"Collection only; existing run_manifest.json and failure.json are preserved."}
        report(ctx,"collection_manifest.json",collection_metadata)
        try:
            collect(ctx)
        except Exception:
            report(ctx,"collection_manifest.json",{**collection_metadata,
                "status":"FAIL","traceback":traceback.format_exc()})
            raise
        return
    report(ctx,"run_manifest.json",run_metadata)
    report(ctx,"failure.json",{"status":"NONE","requested_stage":args.stage})
    try:
        if args.stage in ("preflight","all"):
            preflight(ctx)
        if args.stage in ("parity","benchmark","profile","all"):
            parity(ctx)
        if args.stage in ("benchmark","all"):
            benchmark(ctx)
        if args.stage in ("profile","all"):
            profile(ctx)
        if args.stage in ("collect","all"):
            collect(ctx)
    except Exception:
        report(ctx,"failure.json",{"status":"FAIL","stage":args.stage,"traceback":traceback.format_exc()})
        try:
            collect(ctx)
        except Exception:
            traceback.print_exc()
        raise


if __name__=="__main__":
    main()
