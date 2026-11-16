#!/usr/bin/env python
"""Equal-budget hyperparameter tuning for matched local encoders.

Every encoder receives the same 3 x 2 grid over learning rate and dropout.
The V7 data pipeline, losses, threshold constraints, and evaluation code remain
unchanged.  Use seed 42 for selection, then freeze one configuration per model
family before running final seeds 42, 52, and 62.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
MATCHED_PATH = HERE / "MatchedLocalEncoderBaselines-v1.py"


def load_module():
    if not MATCHED_PATH.exists():
        raise FileNotFoundError(f"Required matched-encoder script not found: {MATCHED_PATH}")
    spec = importlib.util.spec_from_file_location("matched_local_v1", MATCHED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {MATCHED_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MATCHED = load_module()
V7 = MATCHED.V7
ORIGINAL_TRAIN_ONE = V7.train_one

FAMILY_NAMES = {
    "Matched_SelectiveEDRN": "EDRN",
    "Matched_ByteCNN": "ByteCNN",
    "Matched_BiGRU": "BiGRU",
    "Matched_BiLSTM": "BiLSTM",
    "Matched_TinyTransformer": "TinyTransformer",
}
LEARNING_RATES = (5e-4, 1e-3, 2e-3)
DROPOUTS = (0.10, 0.20)


def number_tag(value: float):
    if value < 0.01:
        return f"{value:.0e}".replace("-0", "-")
    return f"{value:.2f}".replace(".", "p")


def build_configs():
    configs = []
    for base in MATCHED.MATCHED_CONFIGS:
        family = FAMILY_NAMES[base["name"]]
        for learning_rate in LEARNING_RATES:
            for dropout in DROPOUTS:
                config = dict(base)
                config.update(
                    {
                        "name": (
                            f"Fair_{family}_lr{number_tag(learning_rate)}_do{number_tag(dropout)}"
                        ),
                        "dropout": dropout,
                        "optimizer_lr": learning_rate,
                        "tuning_family": family,
                        "tuning_protocol": "equal_grid_3lr_x_2dropout_seed42",
                    }
                )
                configs.append(config)
    return configs


FAIR_CONFIGS = build_configs()


def load_fair_configs(value):
    presets = {config["name"]: config for config in FAIR_CONFIGS}
    if value == "fair_tuning":
        return [dict(config) for config in FAIR_CONFIGS]
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(presets))
    if unknown:
        raise ValueError(
            f"Unknown configs {unknown}; use fair_tuning or one of {sorted(presets)}"
        )
    return [dict(presets[name]) for name in names]


def train_one_with_config_lr(config, datasets, args, seed, device, run_id):
    original_lr = args.lr
    args.lr = float(config.get("optimizer_lr", original_lr))
    try:
        return ORIGINAL_TRAIN_ONE(config, datasets, args, seed, device, run_id)
    finally:
        args.lr = original_lr


def main():
    V7.CompactSelectiveSpanEDRN = MATCHED.MatchedLocalEncoderModel
    V7.load_configs = load_fair_configs
    V7.train_one = train_one_with_config_lr
    MATCHED.torch.backends.cuda.enable_flash_sdp(False)
    MATCHED.torch.backends.cuda.enable_mem_efficient_sdp(False)
    MATCHED.torch.backends.cuda.enable_math_sdp(True)
    V7.main()


if __name__ == "__main__":
    main()
