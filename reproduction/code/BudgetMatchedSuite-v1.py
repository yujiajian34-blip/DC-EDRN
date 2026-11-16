#!/usr/bin/env python
"""Parameter-budget matched local-encoder experiments.

The data pipeline, dual-resolution input, pooling, auxiliary heads, losses,
validation constraints, optimizer, and final evaluation protocol are fixed.
Only the local sequence encoder and the width required to meet each parameter
budget are changed.  This is a confirmatory comparison, not a test-set search.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
MATCHED_PATH = HERE / "MatchedLocalEncoderBaselines-v1.py"


def load_module():
    if not MATCHED_PATH.exists():
        raise FileNotFoundError(
            f"Required script not found: {MATCHED_PATH}. Copy this file into the "
            "same code directory as MatchedLocalEncoderBaselines-v1.py."
        )
    spec = importlib.util.spec_from_file_location("budget_matched_local_v1", MATCHED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {MATCHED_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MATCHED = load_module()
V7 = MATCHED.V7
ORIGINAL_TRAIN_ONE = V7.train_one


BUDGETS = {
    "55K": {
        "target": 55_000,
        "word_vocab": 2048,
        "word_dim": 12,
        "semantic_dim": 16,
        "byte_dim": 16,
        "model_dim": 32,
        "rank": 8,
        "units": {
            "DC_EDRN": 76,
            "Full_EDRN": 72,
            "ByteCNN": 36,
            "BiGRU": 36,
            "BiLSTM": 36,
            "TinyTransformer": 52,
        },
    },
    "115K": {
        "target": 115_000,
        "word_vocab": 4096,
        "word_dim": 16,
        "semantic_dim": 24,
        "byte_dim": 24,
        "model_dim": 48,
        "rank": 16,
        "units": {
            "DC_EDRN": 88,
            "Full_EDRN": 84,
            "ByteCNN": 48,
            "BiGRU": 56,
            "BiLSTM": 56,
            "TinyTransformer": 56,
        },
    },
    "350K": {
        "target": 350_000,
        "word_vocab": 8192,
        "word_dim": 32,
        "semantic_dim": 48,
        "byte_dim": 32,
        "model_dim": 64,
        "rank": 24,
        "units": {
            "DC_EDRN": 116,
            "Full_EDRN": 112,
            "ByteCNN": 64,
            "BiGRU": 76,
            "BiLSTM": 76,
            "TinyTransformer": 84,
        },
    },
}


ENCODERS = {
    "DC_EDRN": "selective_edrn",
    "Full_EDRN": "selective_edrn",
    "ByteCNN": "bytecnn",
    "BiGRU": "bigru",
    "BiLSTM": "bilstm",
    "TinyTransformer": "tiny_transformer",
}


def build_configs():
    configs = []
    for budget_name, budget in BUDGETS.items():
        for family, local_encoder in ENCODERS.items():
            config = dict(V7._V7_MEDIUM)
            config.update(
                {
                    "name": f"Budget{budget_name}_{family}",
                    "experiment_family": "parameter_budget_matched_v1",
                    "budget_name": budget_name,
                    "budget_target": int(budget["target"]),
                    "local_encoder": local_encoder,
                    "byte_dim": int(budget["byte_dim"]),
                    "model_dim": int(budget["model_dim"]),
                    "units": int(budget["units"][family]),
                    "rank": int(budget["rank"]),
                    "pool_heads": 1,
                    "dropout": 0.20,
                    "word_vocab": int(budget["word_vocab"]),
                    "word_dim": int(budget["word_dim"]),
                    "semantic_dim": int(budget["semantic_dim"]),
                    "optimizer_lr": 2e-3,
                    "selection_protocol": "fixed_lr_dropout_three_seed_confirmatory",
                }
            )
            if family == "DC_EDRN":
                config.update({"d_max": 1, "selective_duration": False})
            else:
                config.update({"d_max": 8, "selective_duration": True})
            configs.append(config)
    return configs


BUDGET_CONFIGS = build_configs()


def load_configs(value):
    presets = {config["name"]: config for config in BUDGET_CONFIGS}
    aliases = {
        "budget55": [name for name in presets if name.startswith("Budget55K_")],
        "budget115": [name for name in presets if name.startswith("Budget115K_")],
        "budget350": [name for name in presets if name.startswith("Budget350K_")],
        "all_budgets": list(presets),
    }
    if value in aliases:
        return [dict(presets[name]) for name in aliases[value]]
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(presets))
    if unknown:
        raise ValueError(
            f"Unknown configs {unknown}; use budget55, budget115, budget350, "
            f"all_budgets, or one of {sorted(presets)}"
        )
    return [dict(presets[name]) for name in names]


def train_one(config, datasets, args, seed, device, run_id):
    original_lr = args.lr
    args.lr = float(config.get("optimizer_lr", original_lr))
    try:
        return ORIGINAL_TRAIN_ONE(config, datasets, args, seed, device, run_id)
    finally:
        args.lr = original_lr


def print_config_table():
    rows = []
    for config in BUDGET_CONFIGS:
        model = MATCHED.MatchedLocalEncoderModel(config, downsample=4)
        rows.append(
            {
                "Model": config["name"],
                "BudgetTarget": config["budget_target"],
                "TrainParams": V7.count_params(model),
                "InferenceParams": V7.inference_params(model),
                "WordVocab": config["word_vocab"],
                "Units": config["units"],
            }
        )
        del model
    print(json.dumps(rows, indent=2))


def main():
    if "--print-configs" in sys.argv:
        print_config_table()
        return
    V7.CompactSelectiveSpanEDRN = MATCHED.MatchedLocalEncoderModel
    V7.load_configs = load_configs
    V7.train_one = train_one
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    V7.main()


if __name__ == "__main__":
    main()
