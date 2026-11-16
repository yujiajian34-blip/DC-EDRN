# Local release validation, 2026-09-19

The following checks were performed during preparation, without retraining or rerunning the GPU experiments:

- All 18 primary checkpoints: checksum verification, safe weights-only load, strict parameter loading and one synthetic-text forward pass per checkpoint on CPU.
- JSONL inference example: two predictions emitted using the stored threshold, without echoing input text.
- Archived result checker: 24 checks, including the DC-EDRN mean/seed SD, available row-level F1 reconstructions, Table 6 latency values, 3,827-row archived parity, and mean worker summaries.
- Systems data preparation: 3,827 ordered row IDs, text hashes, labels and splits matched the private original benchmark. Only the four inference fields were written to a separate private test directory.

CPU environment: Python 3.12.12, torch 2.9.1 CPU, numpy 2.3.5, pandas 2.3.3, scikit-learn 1.7.2, tqdm 4.67.1, pyarrow 21.0.0. These are observed local versions, not a complete dependency lock or the original GPU environment.

The existing mixed Conda/user-site environment initially failed with duplicate OpenMP runtimes. Validation used `MKL_THREADING_LAYER=SEQUENTIAL` for the checking process only. No global environment was changed and the duplicate-runtime bypass was not enabled. For reuse, create a clean environment as described in README.md; this preparation does not establish clean-install reproducibility.

The systems CPU smoke exposed a Windows non-ASCII profiler output-path failure. The candidate now exports via an ASCII relative temporary path, validates the JSON, and copies the payload using pathlib. Source identity and the packaging fix are recorded in `provenance/systems_sources.json`. This affects diagnostic trace saving, not model math or the timing interval. The current run's completion and trace validation are recorded in `validation/release_checks_20260919.json`.

No CPU timing in these checks should replace the archived 3090 measurements. No complete public download-to-training workflow, CUDA rerun, or newly created isolated dependency environment was validated.
