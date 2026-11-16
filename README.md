# DC-EDRN

Research artifact for **DC-EDRN: Benign-Aware Prompt-Injection Screening for LLM Services**, by Jiajian Yu and Bailin Xie, to appear at IEEE ISPA 2026.

Code, data-construction and evaluation scripts, splits, and checkpoints are released under the [MIT License](LICENSE). Third-party prompt text is not included. Source identities, hashes, and reconstruction instructions are in [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md); redistribution of upstream text follows the source terms. See [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Quick start

Use Python 3.12 in a fresh environment. From this directory:

```sh
python -m pip install -r requirements.txt
python tools/verify_manifest.py
python -m runtime.main --preflight --output local_runs/preflight_cpu.json
python -m runtime.main --text "Please summarize this public example text."
python -m runtime.main --input runtime/examples/texts.jsonl --output local_runs/predictions.jsonl
```

`--checkpoint DC_EDRN:42` is the default; `--list-checkpoints` lists 18 historical checkpoints across six configurations and seeds 42/52/62. The inference command performs no training, calibration, or downloads. Output includes score, checkpoint threshold, binary prediction (`1` = attack) and byte/word truncation flags. Inputs use at most 512 UTF-8 bytes and 128 lexical tokens. Scores are not calibrated risk probabilities.

The portable runtime retains the original auxiliary training head for strict checkpoint loading (323,754 DC-EDRN parameters). The systems implementation removes that head for inference (323,271 parameters); these counts describe different execution wrappers around the same trained model. The CLI applies the stored original threshold, without character normalization or the supplement's filtered-validation recalibration. It is not the optimized latency benchmark entry point.

## Contents and evidence

| Directory | Contents |
|---|---|
| `runtime/` | Portable FP32 inference, configuration and locally staged original checkpoints |
| `reproduction/code/` | Historical construction, training and evaluation scripts; inputs must be obtained separately |
| `data/` | Row IDs, hashes, exclusions and split protocol; no source dataset rows |
| `systems/` | Final GPU profiling/precomputation and timer-corrected live worker scripts |
| `results/` | Archived text-free metrics and per-row scores, including final 2026-09-19 GPU results |
| `validation/` | Historical CPU queue simulation; not the paper's final worker measurement |
| `provenance/` | Source identities and documented packaging transformations |

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), [MODEL_CARD.md](docs/MODEL_CARD.md) and [PAPER_RESULTS.md](docs/PAPER_RESULTS.md). The candidate excludes raw/derived benchmark datasets, manuscript files, submission records, machine caches and original experiment ZIP files. Handwritten construction templates and small synthetic usage examples remain in source code and examples.

## Reproducibility status

The main three-seed comparison uses 885 audit-filtered diagnostic-test rows with original checkpoint selection and thresholds. Historical selection used the original 941-row validation set. The fixed-seed supplement holds weights fixed and recalibrates thresholds on the 880-row filtered validation set plus unchanged benign validation challenges. Filtering does not retroactively change checkpoint selection.

Training data are not bundled. The historical NeurAlchemy download revision remains unknown; the documented card revision must not be substituted as the historical download revision. Hash-checked local reconstruction and archived numerical review are supported; a verified download-to-training reproduction from a clean machine has not been completed.

The model was evaluated as a screening component, including poor BIPIA cross-domain transfer. The systems study is one local GPU worker with synthetic arrivals; it includes neither network nor downstream LLM execution or distributed scaling.
