#!/usr/bin/env python
"""Publication ablations for the frozen Fair EDRN configuration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FAIR_PATH = HERE / "FairMatchedLocalTuning-v2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fair_matched_v2", FAIR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {FAIR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FAIR = load_module()
V7 = FAIR.V7
BASE = next(config for config in FAIR.FAIR_CONFIGS if config["name"] == "Fair_EDRN_lr2e-3_do0p20")


def variant(name, **overrides):
    config = dict(BASE)
    config.update(overrides)
    config["name"] = name
    config["ablation_of"] = BASE["name"]
    return config


ABLATIONS = [
    variant("Paper_Full_EDRN"),
    variant("Paper_NoExplicitDuration", d_max=1, selective_duration=False),
    variant("Paper_StaticDuration", selective_duration=False),
    variant("Paper_NoWordResolution", semantic_context=False),
    variant(
        "Paper_NoSpanEvidence",
        use_span_evidence=False,
        robust_span=False,
        gate_fusion="intent_log_odds",
    ),
    variant("Paper_NoIntentAux", intent_aux=False, gate_fusion="span_only"),
    variant("Paper_NoResidual", residual_fusion=False),
]


def load_configs(value):
    presets = {config["name"]: config for config in ABLATIONS}
    if value == "paper_ablation":
        return [dict(config) for config in ABLATIONS]
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(presets))
    if unknown:
        raise ValueError(f"Unknown ablations {unknown}; available={sorted(presets)}")
    return [dict(presets[name]) for name in names]


def train_one(config, datasets, args, seed, device, run_id):
    original_lr = args.lr
    args.lr = float(config.get("optimizer_lr", original_lr))
    try:
        return FAIR.ORIGINAL_TRAIN_ONE(config, datasets, args, seed, device, run_id)
    finally:
        args.lr = original_lr


def main():
    V7.CompactSelectiveSpanEDRN = FAIR.MATCHED.MatchedLocalEncoderModel
    V7.load_configs = load_configs
    V7.train_one = train_one
    FAIR.MATCHED.torch.backends.cuda.enable_flash_sdp(False)
    FAIR.MATCHED.torch.backends.cuda.enable_mem_efficient_sdp(False)
    FAIR.MATCHED.torch.backends.cuda.enable_math_sdp(True)
    V7.main()


if __name__ == "__main__":
    main()
