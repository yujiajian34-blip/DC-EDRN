# Third-party sources and notices

Reviewed 2026-09-18. These notes identify source evidence. They do not relicense upstream text, and no upstream prompt text is in this repository. See [license status](../LICENSE_STATUS.md).

## Model source files

The following hashes identify the inspected historical originals, not every packaged copy. Portable default-path edits are recorded in runtime/source_manifest.json and provenance/reproduction_sources.json:

| File | SHA-256 |
|---|---|
| `DualResolutionCGEDRN-v7-search.py` | `faf867ccc945ad9d047336c56f5e25e93d353f65a943d22800443c28a9ab19aa` |
| `MatchedLocalEncoderBaselines-v1.py` | `24a9f39d581ae66a4f3f00a59583ee6149f6721cfb3c474238fac0927f69ae43` |

No applicable LICENSE or copyright notice was found in these files or the inspected local code/package roots. The directory name `vendor` does not establish third-party ownership or permission. Authorship, redistribution rights and release terms remain for the authors to determine.

## NeurAlchemy compilation

The reviewed [dataset-card snapshot](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset/blob/58d707d9e36adc17c2321c3375114711d3632322/README.md) declares Apache-2.0 but lists heterogeneous upstream terms: NeurAlchemy originals and hard negatives (Apache-2.0), HackAPrompt (CC BY 4.0), WildGuard/JudgeComparison (Research), and HarmBench (MIT). This is the compilation card's description, not a verified license determination for each local row.

The [official core file tree at that snapshot](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset/tree/58d707d9e36adc17c2321c3375114711d3632322/core) lists:

| Local raw-file alias in the historical summary | Repository-relative path |
|---|---|
| `core_train.parquet` | `core/train-00000-of-00001.parquet` |
| `core_validation.parquet` | `core/validation-00000-of-00001.parquet` |
| `core_test.parquet` | `core/test-00000-of-00001.parquet` |

This is a split/path correspondence, not a historical byte-identity claim. The tree uses `validation-*`; the reviewed card's `val-*` example is inconsistent with those filenames. No dataset was downloaded for this review. **The historical NeurAlchemy download commit is unrecorded.** The snapshot above identifies documentation and paths reviewed, not a newly inferred historical version.

Current upstream evidence also requires source-specific reconciliation:

- [HackAPrompt's official card](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md) declares MIT, differing from the compilation card's CC BY 4.0 label.
- [WildGuardMix's official card](https://huggingface.co/datasets/allenai/wildguardmix/blob/main/README.md?code=true) declares ODC-By and includes research-purpose acceptance under AI2 Responsible Use Guidelines. This does not establish the terms of every row labeled WildGuard/JudgeComparison.
- [HarmBench's official license](https://github.com/centerforaisafety/HarmBench/blob/main/LICENSE) is MIT.

These current pages do not retrospectively settle which upstream versions supplied the local benchmark. Do not present the mixed benchmark as uniformly Apache-2.0.

## NotInject

The historical v2 benchmark summary and builder record revision `66de06ebdf776bf2a53a0bec22a05694b61f362b`. The [official card at that revision](https://huggingface.co/datasets/leolee99/NotInject/blob/66de06ebdf776bf2a53a0bec22a05694b61f362b/README.md) declares MIT. The dataset's official project link resolves to PIGuard; its [MIT license](https://github.com/leolee99/PIGuard/blob/main/LICENSE) carries `Copyright (c) 2024 Hao Li` and requires preservation of the copyright and permission notices when applicable.

Attribute NotInject to the upstream dataset and associated InjecGuard/PIGuard work. Any later redistribution of identifiable NotInject-derived rows should retain applicable notices and document transformations. No NotInject prompt text is included in this candidate.

## Recovered archive evidence

The preserved reproduction-runtime archive contains the original benchmark with SHA-256 `2c412a16ae445df31b4e162860d1b3980f58b53ae12b62b9995860690bb950e7`, exactly matching the supplement's original-benchmark manifest. A separate current v3 parquet has different serialized bytes; the preparation audit found all 34,116 rows and all 25 columns equal under pandas comparison. Preserve the archived hash as the reference identity.

The v3 summary links to v2 SHA-256 `60b2cbe086433ba9890bc6ce97a5725a2b4dd0cd753cf0e9e082476e2439cb16`, which matches both inspected v2 backups. This recovers local archive continuity and the recorded NotInject revision. It does not recover the missing NeurAlchemy download commit or resolve source-specific redistribution rights.

Raw and derived prompt text stay out of this repository. Project code and checkpoints are under the repository MIT License; this notice does not relicense upstream datasets.
