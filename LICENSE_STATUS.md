# License status

Status: **public release**, 2026-11-16, at https://github.com/yujiajian34-blip/DC-EDRN.

The project code, documentation, split manifests, text-free results, and project-trained checkpoints in this repository are released under the MIT License, copyright 2026 Jiajian Yu and Bailin Xie. That grant does not relicense third-party material.

| Material | Terms |
|---|---|
| Project scripts, configurations, documentation, and text-free results | MIT License |
| `DualResolutionCGEDRN-v7-search.py` and `MatchedLocalEncoderBaselines-v1.py` | Included as project source under the MIT License. No separate upstream license was present in the inspected copies |
| Project-trained checkpoints | MIT License, same copyright. Training-data obligations remain with the upstream sources |
| Third-party datasets and derived prompt text | Not in this repository. See [third-party notices](docs/THIRD_PARTY_NOTICES.md) |
| Installed third-party software | Governed by its upstream terms; not bundled |

Dataset-card metadata does not grant a blanket license for the mixed benchmark, project code or checkpoints. Source labels, hashes, split descriptions and text-free aggregate measurements may be documented without including the source prompts.

Keep all raw and derived prompt-text artifacts outside `public_release/`, including parquet datasets, text-bearing row exports and internal transfer archives. A complete private reproducibility archive is a separate artifact.

See [third-party notices](docs/THIRD_PARTY_NOTICES.md) for reviewed sources, known version evidence and remaining limitations. The public repository contains code, construction and evaluation scripts, splits, and checkpoints. It does not contain upstream prompt text.
