# Reproducibility guide

## 1. Inspect archived results without data or a GPU

```sh
python tools/verify_manifest.py
python tools/check_results.py
```

These checks verify file integrity and selected paper aggregates; they do not rerun model evaluation. SHA-256 establishes consistency with this local manifest, not external authenticity.

## 2. Inference on your own text

Follow README.md. CPU checks load all 18 original weights with `weights_only=True`, strict parameter loading, checksum verification and one synthetic input per model. This is a workflow check, not benchmark reproduction. Current direct dependency ranges are declared, not a fully resolved cross-platform lockfile. Recorded environments remain in archived validation/result files.

## 3. Local data reconstruction

Obtain upstream data under its applicable terms. `tools/fetch_core.py --revision REVISION --output LOCAL_DIRECTORY` checks downloaded core files against archived byte hashes and refuses mismatches. The historical core download revision is unrecorded; no revision is asserted to reproduce those bytes. `reproduction/code/build_edrn_benchmark_v2.py` and `build_context_gated_benchmark_v3.py` preserve historical construction logic; use each script's `--help` and supply explicit local input/output paths.

For an existing legally obtained original benchmark:

```sh
python tools/validate_data.py --file /your/local/benchmark_original.parquet
python tools/prepare_systems_data.py --file /your/local/benchmark_original.parquet
```

The second command selects the exact 3,827 ordered evaluation IDs and checks labels, splits and text hashes. It writes only the four fields needed for performance inference to ignored `systems/data/evaluation.parquet`. Serialized bytes differ from the historical all-column parquet; input identity is established by row hashes. No threshold fitting or training takes place. Do not commit generated datasets.

## 4. Systems workflow

Once local data are prepared, from the candidate root:

```sh
python systems/code/measure.py --stage all --device cpu --smoke
```

The CPU smoke is never a reproduction of RTX 3090 latency. On an appropriate GPU, formal commands are:

```sh
python systems/code/measure.py --stage all --device cuda
python systems/code/live_gateway_gpu.py --device cuda --rates 10 20 40 80 --patterns periodic bursty --duration 20 --repeats 3 --warmup 10
```

Install an appropriate CUDA PyTorch build before GPU runs. Formal measurements create `systems/results/` separately from archived `results/`. The gateway script requires a successful formal parity run first. Changed software/hardware/input serialization yields different fingerprints and timing; preserve new manifests and do not replace the archived paper evidence.

## 5. Training and full evaluation

Historical training/evaluation entry points are included in `reproduction/code/`. Some require optional packages (such as transformers, scipy or huggingface_hub), separately obtained guard-model snapshots, reconstructed labels/spans and local path arguments. They are reference scripts, not an end-to-end validated public training pipeline. The row-text manifest does not validate all auxiliary training targets. Full retraining, optional public-guard execution, raw-data download reconstruction and GPU reruns were not performed for this release preparation.
