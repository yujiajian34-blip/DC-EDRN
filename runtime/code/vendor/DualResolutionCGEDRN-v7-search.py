import argparse
import ast
import gc
import json
import math
import os
import random
import re
import time
import zlib
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


DEFAULT_DATA_FILE = Path("data/neuralchemy_context_gated_benchmark_v3.parquet")
DEFAULT_SAVE_DIR = Path("results/dual_resolution_cgedrn_v7")
PAD_ID = 0
BYTE_VOCAB_SIZE = 257
DURATION_TO_ID = {"short": 0, "medium": 1, "long": 2}

PRESET_CONFIGS = [
    {
        "name": "CSED-RN_XS1",
        "byte_dim": 24,
        "model_dim": 48,
        "units": 48,
        "d_max": 4,
        "rank": 16,
        "kernels": [3, 5],
        "pool_heads": 2,
        "dropout": 0.15,
    },
    {
        "name": "CSED-RN_S1",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 64,
        "d_max": 4,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.15,
    },
    {
        "name": "CSED-RN_S2",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 64,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
    },
    {
        "name": "CSED-RN_S2_StaticDuration",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 64,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "selective_duration": False,
    },
    {
        "name": "CSED-RN_S3",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
    },
    {
        "name": "CSED-RN_M1",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 96,
        "d_max": 8,
        "rank": 32,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
    },
    {
        "name": "CSED-RN_M2",
        "byte_dim": 40,
        "model_dim": 80,
        "units": 96,
        "d_max": 12,
        "rank": 32,
        "kernels": [3, 5, 9],
        "pool_heads": 3,
        "dropout": 0.20,
    },
    {
        "name": "CSED-RN_M3",
        "byte_dim": 40,
        "model_dim": 80,
        "units": 128,
        "d_max": 8,
        "rank": 40,
        "kernels": [3, 5, 9],
        "pool_heads": 3,
        "dropout": 0.20,
    },
    {
        "name": "CSED-RN_L1",
        "byte_dim": 48,
        "model_dim": 96,
        "units": 128,
        "d_max": 12,
        "rank": 48,
        "kernels": [3, 5, 9, 13],
        "pool_heads": 3,
        "dropout": 0.20,
    },
    {
        "name": "CSED-RN_L2",
        "byte_dim": 48,
        "model_dim": 112,
        "units": 160,
        "d_max": 16,
        "rank": 64,
        "kernels": [3, 5, 9, 13],
        "pool_heads": 4,
        "dropout": 0.20,
    },
]


IMPROVED_CONFIGS = [
    {
        "name": "LEDRN_XS_FusedEvidence",
        "byte_dim": 24,
        "model_dim": 48,
        "units": 48,
        "d_max": 4,
        "rank": 16,
        "kernels": [3, 5],
        "pool_heads": 2,
        "dropout": 0.15,
        "pool_type": "fused",
        "robust_span": True,
    },
    {
        "name": "LEDRN_XS_SharedBi",
        "byte_dim": 24,
        "model_dim": 48,
        "units": 48,
        "d_max": 4,
        "rank": 16,
        "kernels": [3, 5],
        "pool_heads": 2,
        "dropout": 0.15,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
    },
    {
        "name": "LEDRN_XS_SharedResidual",
        "byte_dim": 24,
        "model_dim": 48,
        "units": 48,
        "d_max": 4,
        "rank": 16,
        "kernels": [3, 5],
        "pool_heads": 2,
        "dropout": 0.15,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "residual_fusion": True,
    },
    {
        "name": "LEDRN_S_FusedEvidence",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 64,
        "d_max": 4,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.15,
        "pool_type": "fused",
        "robust_span": True,
    },
    {
        "name": "LEDRN_S_SharedResidual_D4",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 64,
        "d_max": 4,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.15,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "residual_fusion": True,
    },
    {
        "name": "LEDRN_S_SharedResidual_D8",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 64,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "residual_fusion": True,
    },
    {
        "name": "LEDRN_S_IndependentResidual",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 64,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "residual_fusion": True,
    },
    {
        "name": "LEDRN_SWide_SharedResidual",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "residual_fusion": True,
    },
]


ADVANCED_CONFIGS = [
    {
        "name": "AEDRN_SWide_CurrentWinner",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "residual_fusion": True,
    },
    {
        "name": "AEDRN_SWide_NoResidual",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
    },
    {
        "name": "AEDRN_SWide_ChannelResidual",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "residual_fusion": True,
        "residual_gate_type": "channel",
    },
    {
        "name": "AEDRN_SWide_ResidualSwiGLU",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "residual_fusion": True,
        "head_type": "swiglu",
    },
    {
        "name": "AEDRN_SWide_ProgressiveDuration",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "residual_fusion": True,
        "duration_prior": "progressive",
    },
    {
        "name": "AEDRN_SWide_D4",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 4,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "residual_fusion": True,
    },
    {
        "name": "AEDRN_SWide_MaxSpan",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": False,
        "shared_bidir": True,
        "residual_fusion": True,
    },
    {
        "name": "AEDRN_SWide_IndependentBi",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "residual_fusion": True,
    },
    {
        "name": "AEDRN_SWide_DirectionFuse",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "direction_fusion": "gated_sum",
    },
    {
        "name": "AEDRN_S_DirectionFuse",
        "byte_dim": 32,
        "model_dim": 64,
        "units": 64,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "direction_fusion": "gated_sum",
    },
    {
        "name": "AEDRN_XS_DirectionFuse",
        "byte_dim": 24,
        "model_dim": 48,
        "units": 48,
        "d_max": 4,
        "rank": 16,
        "kernels": [3, 5],
        "pool_heads": 2,
        "dropout": 0.15,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "direction_fusion": "gated_sum",
    },
]

CONTEXT_CONFIGS = [
    {
        "name": name,
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "direction_fusion": "concat",
        "residual_fusion": True,
        "residual_gate_type": "scalar",
        "context_gate": gate,
        "intent_aux": gate,
        "use_context_data": context_data,
        "use_pair_loss": pair_loss,
    }
    for name, gate, context_data, pair_loss in [
        ("CGEDRN_Baseline", False, False, False),
        ("CGEDRN_DataOnly", False, True, True),
        ("CGEDRN_GateOnly", True, False, False),
        ("CGEDRN_Full", True, True, True),
    ]
]

V5_CONFIGS = [
    {
        "name": name,
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "direction_fusion": "concat",
        "residual_fusion": True,
        "residual_gate_type": "scalar",
        "context_gate": gate_mode in {"logit_residual", "dual"},
        "intent_aux": gate_mode in {"intent_log_odds", "dual"},
        "gate_fusion": gate_mode,
        "use_span_evidence": use_span,
        "use_context_data": True,
        "use_pair_loss": use_pair,
        "pair_loss_weight": 0.03,
    }
    for name, gate_mode, use_span, use_pair in [
        ("CGEDRN_DirectGate", "logit_residual", True, False),
        ("CGEDRN_DirectGate_NoSpan", "logit_residual", False, False),
        ("CGEDRN_IntentFusion", "intent_log_odds", True, False),
        ("CGEDRN_DualContext", "dual", True, False),
        ("CGEDRN_DualContext_WeakPair", "dual", True, True),
    ]
]

V6_CONFIGS = [
    {
        "name": name,
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "direction_fusion": "concat",
        "residual_fusion": True,
        "residual_gate_type": "scalar",
        "context_gate": fusion == "dual",
        "intent_aux": True,
        "gate_fusion": fusion,
        "use_span_evidence": use_span,
        "use_context_data": True,
        "use_pair_loss": use_pair,
        "pair_loss_weight": 0.03,
    }
    for name, fusion, use_span, use_pair in [
        ("CGEDRN_IntentNoSpan", "intent_log_odds", False, False),
        ("CGEDRN_IntentGatedSpan", "intent_gated_span", True, False),
        ("CGEDRN_IntentOnly", "intent_only", False, False),
        ("CGEDRN_DualNoSpan", "dual", False, False),
        ("CGEDRN_IntentGatedSpan_WeakPair", "intent_gated_span", True, True),
    ]
]

V7_CONFIGS = [
    {
        "name": name,
        "byte_dim": 32,
        "model_dim": 64,
        "units": 80,
        "d_max": 8,
        "rank": 24,
        "kernels": [3, 5, 9],
        "pool_heads": 2,
        "dropout": 0.20,
        "pool_type": "fused",
        "robust_span": True,
        "shared_bidir": True,
        "direction_fusion": "concat",
        "residual_fusion": True,
        "residual_gate_type": "scalar",
        "context_gate": False,
        "intent_aux": True,
        "gate_fusion": fusion,
        "use_span_evidence": use_span,
        "use_context_data": True,
        "use_pair_loss": False,
        "semantic_context": True,
        "word_vocab": 8192,
        "word_dim": word_dim,
        "semantic_dim": semantic_dim,
        "semantic_kernels": [3, 5],
    }
    for name, word_dim, semantic_dim, fusion, use_span in [
        ("DRCGEDRN_Tiny", 16, 24, "intent_gated_span", True),
        ("DRCGEDRN_Small", 24, 32, "intent_gated_span", True),
        ("DRCGEDRN_Small_NoSpan", 24, 32, "intent_log_odds", False),
        ("DRCGEDRN_Small_IntentOnly", 24, 32, "intent_only", False),
        ("DRCGEDRN_Medium", 32, 48, "intent_gated_span", True),
    ]
]

_V7_MEDIUM = next(config for config in V7_CONFIGS if config["name"] == "DRCGEDRN_Medium")
V7_ABLATION_CONFIGS = []
for _name, _overrides in [
    ("DRCGEDRN_Medium_NoSpanFusion", {"gate_fusion": "intent_log_odds", "use_span_evidence": False}),
    ("DRCGEDRN_Medium_StaticDuration", {"selective_duration": False}),
    ("DRCGEDRN_Medium_Dmax4", {"d_max": 4}),
    ("DRCGEDRN_Medium_NoResidual", {"residual_fusion": False}),
]:
    _config = dict(_V7_MEDIUM)
    _config.update(_overrides)
    _config["name"] = _name
    V7_ABLATION_CONFIGS.append(_config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Context-gated compact EDRN with constrained low-FPR selection."
    )
    parser.add_argument("--mode", choices=["search", "final", "smoke"], default="search")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--configs", default="dual_resolution")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-bytes", type=int, default=512)
    parser.add_argument("--max-words", type=int, default=128)
    parser.add_argument("--word-vocab", type=int, default=8192)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--span-loss-weight", type=float, default=0.25)
    parser.add_argument("--duration-loss-weight", type=float, default=0.10)
    parser.add_argument("--intent-loss-weight", type=float, default=0.15)
    parser.add_argument("--gate-loss-weight", type=float, default=0.10)
    parser.add_argument("--pair-loss-weight", type=float, default=0.15)
    parser.add_argument("--pair-margin", type=float, default=1.0)
    parser.add_argument("--pair-every", type=int, default=4)
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--max-fpr-notinject", type=float, default=0.10)
    parser.add_argument("--max-fpr-overdefense", type=float, default=0.15)
    parser.add_argument("--span-pos-weight", type=float, default=6.0)
    parser.add_argument("--fgm", action="store_true")
    parser.add_argument("--fgm-epsilon", type=float, default=0.25)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-params", type=int, default=500000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--eval-splits",
        default="diagnostic_test,diagnostic_notinject,diagnostic_obfuscation,diagnostic_long_context,diagnostic_overdefense",
    )
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip Model/Seed pairs already present in search_runs.csv.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(choice)


def load_configs(value):
    presets = {
        cfg["name"]: cfg
        for cfg in PRESET_CONFIGS + IMPROVED_CONFIGS + ADVANCED_CONFIGS + CONTEXT_CONFIGS + V5_CONFIGS + V6_CONFIGS + V7_CONFIGS + V7_ABLATION_CONFIGS
    }
    if value == "preset":
        return [dict(cfg) for cfg in PRESET_CONFIGS]
    if value == "improved":
        return [dict(cfg) for cfg in IMPROVED_CONFIGS]
    if value == "advanced":
        return [dict(cfg) for cfg in ADVANCED_CONFIGS]
    if value == "context_ablation":
        return [dict(cfg) for cfg in CONTEXT_CONFIGS]
    if value == "context_v2":
        return [dict(cfg) for cfg in V5_CONFIGS]
    if value == "context_v3":
        return [dict(cfg) for cfg in V6_CONFIGS]
    if value == "dual_resolution":
        return [dict(cfg) for cfg in V7_CONFIGS]
    if value == "v7_ablation":
        return [dict(cfg) for cfg in V7_ABLATION_CONFIGS]
    if value.strip().startswith("["):
        configs = ast.literal_eval(value)
    else:
        names = [name.strip() for name in value.split(",") if name.strip()]
        unknown = sorted(set(names) - set(presets))
        if unknown:
            raise ValueError(f"Unknown configs {unknown}; available: {sorted(presets)}")
        configs = [dict(presets[name]) for name in names]
    required = {
        "name",
        "byte_dim",
        "model_dim",
        "units",
        "d_max",
        "rank",
        "kernels",
        "pool_heads",
        "dropout",
    }
    for config in configs:
        missing = required - set(config)
        if missing:
            raise ValueError(f"Config {config.get('name', '<unnamed>')} is missing {sorted(missing)}")
        config.setdefault("selective_duration", True)
    return configs


def load_frame(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_parquet(path)
    required = {
        "text",
        "label",
        "split",
        "span_start",
        "span_end",
        "duration_bucket",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["text", "label", "split"]).copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["span_start"] = frame["span_start"].fillna(-1).astype(int)
    frame["span_end"] = frame["span_end"].fillna(-1).astype(int)
    for split in ["train", "val", "val_notinject", "val_overdefense"]:
        if split not in set(frame["split"]):
            raise ValueError(f"Required split '{split}' is absent")
    return frame.reset_index(drop=True)


WORD_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def encode_frame(frame, max_bytes, max_words, word_vocab):
    count = len(frame)
    input_ids = np.zeros((count, max_bytes), dtype=np.int16)
    word_ids = np.zeros((count, max_words), dtype=np.int32)
    span_targets = np.zeros((count, max_bytes), dtype=np.uint8)
    labels = frame["label"].to_numpy(dtype=np.float32)
    duration_ids = np.full(count, -100, dtype=np.int64)
    weights = (
        frame["sample_weight"].to_numpy(dtype=np.float32)
        if "sample_weight" in frame.columns
        else np.ones(count, dtype=np.float32)
    )
    intent_ids = (
        frame["intent_label"].to_numpy(dtype=np.int64, copy=True)
        if "intent_label" in frame.columns
        else np.where(labels > 0, 2, 0).astype(np.int64)
    )

    for i, row in enumerate(tqdm(frame.itertuples(index=False), total=count, desc="Byte encoding", leave=False)):
        text = str(row.text)
        encoded = text.encode("utf-8")[:max_bytes]
        if encoded:
            input_ids[i, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8).astype(np.int16) + 1
        tokens = WORD_PATTERN.findall(text.casefold())[:max_words]
        for j, token in enumerate(tokens):
            word_ids[i, j] = zlib.crc32(token.encode("utf-8")) % (word_vocab - 1) + 1
        if int(row.label) == 1 and int(row.span_start) >= 0 and int(row.span_end) > int(row.span_start):
            byte_start = len(text[: int(row.span_start)].encode("utf-8"))
            byte_end = len(text[: int(row.span_end)].encode("utf-8"))
            byte_start = min(max_bytes, byte_start)
            byte_end = min(max_bytes, byte_end)
            if byte_end > byte_start:
                span_targets[i, byte_start:byte_end] = 1
                duration_ids[i] = DURATION_TO_ID.get(str(row.duration_bucket), -100)

    return TensorDataset(
        torch.from_numpy(input_ids.astype(np.int64)),
        torch.from_numpy(labels),
        torch.from_numpy(span_targets.astype(np.float32)),
        torch.from_numpy(duration_ids),
        torch.from_numpy(weights),
        torch.from_numpy(intent_ids),
        torch.from_numpy(word_ids.astype(np.int64)),
    )


def encode_all_splits(frame, max_bytes, max_words, word_vocab):
    datasets = {}
    for split, split_frame in frame.groupby("split", sort=False):
        datasets[str(split)] = encode_frame(
            split_frame.reset_index(drop=True), max_bytes, max_words, word_vocab
        )
    return datasets


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def concat_tensor_datasets(*datasets):
    datasets = [dataset for dataset in datasets if dataset is not None and len(dataset)]
    if len(datasets) == 1:
        return datasets[0]
    return TensorDataset(*(torch.cat([dataset.tensors[i] for dataset in datasets]) for i in range(len(datasets[0].tensors))))


def encode_pair_dataset(frame, max_bytes, max_words, word_vocab):
    paired = frame[frame.get("pair_id", pd.Series(index=frame.index, dtype=object)).notna()].copy()
    if paired.empty or "pair_role" not in paired:
        return None
    positive = paired[paired["pair_role"].eq(1)].drop_duplicates("pair_id").set_index("pair_id")
    negative = paired[paired["pair_role"].eq(0)].drop_duplicates("pair_id").set_index("pair_id")
    pair_ids = positive.index.intersection(negative.index).sort_values()
    if len(pair_ids) == 0:
        return None
    positive_encoded = encode_frame(
        positive.loc[pair_ids].reset_index(), max_bytes, max_words, word_vocab
    )
    negative_encoded = encode_frame(
        negative.loc[pair_ids].reset_index(), max_bytes, max_words, word_vocab
    )
    return TensorDataset(
        positive_encoded.tensors[0], positive_encoded.tensors[6],
        negative_encoded.tensors[0], negative_encoded.tensors[6],
    )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class MultiScaleByteStem(nn.Module):
    def __init__(self, byte_dim, model_dim, kernels, stride, dropout):
        super().__init__()
        self.stride = stride
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        byte_dim,
                        byte_dim,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,
                        groups=byte_dim,
                        bias=False,
                    ),
                    nn.Conv1d(byte_dim, model_dim, kernel_size=1, bias=False),
                    nn.GELU(),
                )
                for kernel in kernels
            ]
        )
        self.branch_logits = nn.Parameter(torch.zeros(len(kernels)))
        self.norm = RMSNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, byte_mask):
        x = x.transpose(1, 2)
        branch_weight = torch.softmax(self.branch_logits, dim=0)
        outputs = [branch(x).transpose(1, 2) for branch in self.branches]
        hidden = sum(weight * output for weight, output in zip(branch_weight, outputs))
        mask = F.max_pool1d(
            byte_mask.float().unsqueeze(1),
            kernel_size=self.stride,
            stride=self.stride,
            ceil_mode=True,
        ).squeeze(1).bool()
        hidden = self.dropout(self.norm(hidden)) * mask.unsqueeze(-1)
        return hidden, mask


class LowRankTransition(nn.Module):
    def __init__(self, units, rank):
        super().__init__()
        self.down = nn.Linear(units, rank, bias=False)
        self.up = nn.Linear(rank, units, bias=False)

    def forward(self, x):
        return self.up(F.silu(self.down(x)))


class SelectiveEDRNDirection(nn.Module):
    def __init__(
        self,
        input_dim,
        units,
        d_max,
        rank,
        reverse=False,
        selective_duration=True,
        duration_prior="flat",
    ):
        super().__init__()
        self.units = units
        self.d_max = d_max
        self.reverse = reverse
        self.selective_duration = selective_duration
        self.duration_prior = duration_prior
        self.x_proj = nn.Linear(input_dim, rank)
        self.h_proj = nn.Linear(units, rank, bias=False)
        self.gate_proj = nn.Linear(rank, 4 * units)
        if selective_duration:
            self.duration_delta = nn.Linear(rank, d_max, bias=False)
            self.stay_delta = nn.Linear(rank, d_max, bias=False)
            self.duration_temperature = nn.Linear(rank, units)
        else:
            self.duration_delta = None
            self.stay_delta = None
            self.duration_temperature = None
        self.start_logits = nn.Parameter(torch.zeros(units, d_max))
        self.stay_logits = nn.Parameter(torch.zeros(units, d_max))
        self.boundary_transition = LowRankTransition(units, rank)
        self.out_norm = RMSNorm(units)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.start_logits, mean=0.0, std=0.02)
        if self.duration_prior == "progressive":
            probabilities = torch.linspace(0.35, 0.80, self.d_max)
            logits = torch.logit(probabilities).unsqueeze(0).expand(self.units, -1)
            with torch.no_grad():
                self.stay_logits.copy_(logits)
        elif self.duration_prior == "flat":
            nn.init.constant_(self.stay_logits, 0.5)
        else:
            raise ValueError(f"Unknown duration_prior: {self.duration_prior}")
        nn.init.constant_(self.gate_proj.bias[: self.units], 1.0)
        nn.init.constant_(self.gate_proj.bias[self.units : 2 * self.units], -1.0)

    def forward(self, x, mask, reverse=None):
        batch, seq_len, _ = x.shape
        hidden = x.new_zeros(batch, self.units)
        alpha = x.new_zeros(batch, self.units, self.d_max)
        outputs = [None] * seq_len
        run_reverse = self.reverse if reverse is None else bool(reverse)
        steps = range(seq_len - 1, -1, -1) if run_reverse else range(seq_len)

        for t in steps:
            valid = mask[:, t].view(batch, 1)
            core = F.silu(self.x_proj(x[:, t]) + self.h_proj(hidden))
            keep_logit, reset_logit, candidate, output_gate = self.gate_proj(core).chunk(4, dim=-1)
            mixture = torch.softmax(torch.stack([keep_logit, reset_logit], dim=-1), dim=-1)
            keep = mixture[..., 0].unsqueeze(-1)
            restart = mixture[..., 1].unsqueeze(-1)

            if self.selective_duration:
                temperature = F.softplus(self.duration_temperature(core)).unsqueeze(-1) + 0.5
                start = self.start_logits.unsqueeze(0) * temperature
                start = start + self.duration_delta(core).unsqueeze(1)
                start = torch.softmax(start, dim=-1)
                stay = self.stay_logits.unsqueeze(0) + self.stay_delta(core).unsqueeze(1)
                stay = torch.sigmoid(stay)
            else:
                start = torch.softmax(self.start_logits, dim=-1).unsqueeze(0).expand(batch, -1, -1)
                stay = torch.sigmoid(self.stay_logits).unsqueeze(0).expand(batch, -1, -1)
            propagated = alpha * stay
            propagated[:, :, 1:] = propagated[:, :, 1:] + alpha[:, :, :-1] * (1.0 - stay[:, :, :-1])
            boundary = alpha[:, :, -1] * (1.0 - stay[:, :, -1])
            candidate = torch.tanh(candidate + self.boundary_transition(boundary))
            alpha_new = keep * propagated + restart * candidate.unsqueeze(-1) * start
            hidden_new = torch.sigmoid(output_gate) * torch.tanh(alpha_new.sum(dim=-1))
            hidden_new = self.out_norm(hidden_new)

            alpha = torch.where(valid.unsqueeze(-1), alpha_new, alpha)
            hidden = torch.where(valid, hidden_new, hidden)
            outputs[t] = hidden * valid

        return torch.stack(outputs, dim=1)


class SelectiveBiEDRN(nn.Module):
    def __init__(
        self,
        input_dim,
        units,
        d_max,
        rank,
        selective_duration=True,
        shared_directions=False,
        direction_fusion="concat",
        duration_prior="flat",
    ):
        super().__init__()
        self.shared_directions = shared_directions
        self.direction_fusion = direction_fusion
        if shared_directions:
            self.shared_edrn = SelectiveEDRNDirection(
                input_dim,
                units,
                d_max,
                rank,
                selective_duration=selective_duration,
                duration_prior=duration_prior,
            )
            self.forward_edrn = None
            self.backward_edrn = None
        else:
            self.shared_edrn = None
            self.forward_edrn = SelectiveEDRNDirection(
                input_dim,
                units,
                d_max,
                rank,
                reverse=False,
                selective_duration=selective_duration,
                duration_prior=duration_prior,
            )
            self.backward_edrn = SelectiveEDRNDirection(
                input_dim,
                units,
                d_max,
                rank,
                reverse=True,
                selective_duration=selective_duration,
                duration_prior=duration_prior,
            )
        if direction_fusion == "concat":
            self.output_dim = 2 * units
            self.direction_logits = None
        elif direction_fusion == "gated_sum":
            self.output_dim = units
            self.direction_logits = nn.Parameter(torch.zeros(units))
        else:
            raise ValueError(f"Unknown direction_fusion: {direction_fusion}")
        self.norm = RMSNorm(self.output_dim)

    def forward(self, x, mask):
        if self.shared_directions:
            forward = self.shared_edrn(x, mask, reverse=False)
            backward = self.shared_edrn(x, mask, reverse=True)
        else:
            forward = self.forward_edrn(x, mask)
            backward = self.backward_edrn(x, mask)
        if self.direction_fusion == "concat":
            combined = torch.cat([forward, backward], dim=-1)
        else:
            gate = torch.sigmoid(self.direction_logits).view(1, 1, -1)
            combined = gate * forward + (1.0 - gate) * backward
        return self.norm(combined) * mask.unsqueeze(-1)


class MultiQueryPooling(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(heads, dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.queries, std=dim**-0.5)

    def forward(self, hidden, mask):
        scores = torch.einsum("btd,hd->bht", hidden, self.queries) / math.sqrt(hidden.size(-1))
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bht,btd->bhd", attention, hidden).flatten(1)
        maximum = hidden.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        return self.dropout(torch.cat([attended, maximum], dim=-1))


class FusedEvidencePooling(nn.Module):
    """Compress attention, mean, and max evidence back to one hidden width."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(heads, dim))
        self.source_logits = nn.Parameter(torch.zeros(heads + 2, dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.queries, std=dim**-0.5)

    def forward(self, hidden, mask):
        scores = torch.einsum("btd,hd->bht", hidden, self.queries) / math.sqrt(hidden.size(-1))
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bht,btd->bhd", attention, hidden)
        valid = mask.unsqueeze(-1)
        mean = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        maximum = hidden.masked_fill(~valid, -1e4).max(dim=1).values
        sources = torch.cat([attended, mean.unsqueeze(1), maximum.unsqueeze(1)], dim=1)
        weights = torch.softmax(self.source_logits, dim=0).unsqueeze(0)
        return self.dropout((sources * weights).sum(dim=1))


class SwiGLUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.norm = RMSNorm(input_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        hidden = self.value(x) * F.silu(self.gate(x))
        return self.output(self.dropout(hidden))


class HashedWordContextEncoder(nn.Module):
    def __init__(self, vocab_size, word_dim, semantic_dim, kernels, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(
                word_dim,
                word_dim,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=word_dim,
                bias=False,
            )
            for kernel in kernels
        )
        source_dim = word_dim * (len(kernels) + 1)
        self.projection = nn.Sequential(
            RMSNorm(source_dim),
            nn.Linear(source_dim, semantic_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = semantic_dim * 2

    def forward(self, word_ids):
        mask = word_ids.ne(0)
        embedded = self.embedding(word_ids)
        channels = embedded.transpose(1, 2)
        sources = [embedded]
        for convolution in self.convs:
            sources.append(F.silu(convolution(channels)).transpose(1, 2))
        hidden = self.projection(torch.cat(sources, dim=-1))
        hidden = hidden * mask.unsqueeze(-1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean_pool = hidden.sum(dim=1) / denominator
        max_pool = hidden.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        max_pool = torch.where(mask.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=-1)


class CompactSelectiveSpanEDRN(nn.Module):
    def __init__(self, config, downsample):
        super().__init__()
        byte_dim = int(config["byte_dim"])
        model_dim = int(config["model_dim"])
        units = int(config["units"])
        pool_heads = int(config["pool_heads"])
        dropout = float(config["dropout"])
        self.robust_span = bool(config.get("robust_span", False))
        self.residual_fusion = bool(config.get("residual_fusion", False))
        self.context_gate = bool(config.get("context_gate", False))
        self.intent_aux = bool(config.get("intent_aux", False))
        self.gate_fusion = str(config.get("gate_fusion", "span_only"))
        self.use_span_evidence = bool(config.get("use_span_evidence", True))
        self.intent_inference = self.gate_fusion in {
            "intent_log_odds", "intent_gated_span", "intent_only", "dual"
        }
        self.semantic_context = bool(config.get("semantic_context", False))
        self.downsample = downsample
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, byte_dim, padding_idx=PAD_ID)
        self.stem = MultiScaleByteStem(
            byte_dim,
            model_dim,
            list(config["kernels"]),
            downsample,
            dropout,
        )
        self.edrn = SelectiveBiEDRN(
            model_dim,
            units,
            int(config["d_max"]),
            int(config["rank"]),
            selective_duration=bool(config.get("selective_duration", True)),
            shared_directions=bool(config.get("shared_bidir", False)),
            direction_fusion=str(config.get("direction_fusion", "concat")),
            duration_prior=str(config.get("duration_prior", "flat")),
        )
        hidden_dim = self.edrn.output_dim
        pool_type = str(config.get("pool_type", "multiquery"))
        if pool_type == "fused":
            pool_dim = hidden_dim
            self.pool = FusedEvidencePooling(hidden_dim, pool_heads, dropout)
        elif pool_type == "multiquery":
            pool_dim = hidden_dim * (pool_heads + 1)
            self.pool = MultiQueryPooling(hidden_dim, pool_heads, dropout)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        head_dim = max(32, units)
        if self.semantic_context:
            self.word_encoder = HashedWordContextEncoder(
                int(config.get("word_vocab", 8192)),
                int(config.get("word_dim", 24)),
                int(config.get("semantic_dim", 32)),
                list(config.get("semantic_kernels", [3, 5])),
                dropout,
            )
            context_dim = pool_dim + self.word_encoder.output_dim
        else:
            self.word_encoder = None
            context_dim = pool_dim
        if self.residual_fusion:
            self.residual_proj = nn.Linear(model_dim, hidden_dim, bias=False)
            residual_gate_type = str(config.get("residual_gate_type", "scalar"))
            if residual_gate_type == "scalar":
                self.residual_gate = nn.Parameter(torch.tensor(-1.0))
            elif residual_gate_type == "channel":
                self.residual_gate = nn.Parameter(torch.full((hidden_dim,), -1.0))
            else:
                raise ValueError(f"Unknown residual_gate_type: {residual_gate_type}")
            self.residual_norm = RMSNorm(hidden_dim)
        else:
            self.residual_proj = None
            self.residual_gate = None
            self.residual_norm = None
        head_type = str(config.get("head_type", "gelu"))
        if head_type == "gelu":
            self.classifier = nn.Sequential(
                RMSNorm(pool_dim),
                nn.Linear(pool_dim, head_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, 1),
            )
        elif head_type == "swiglu":
            self.classifier = SwiGLUClassifier(pool_dim, head_dim, dropout)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        self.span_head = nn.Linear(hidden_dim, 1)
        self.duration_head = nn.Linear(pool_dim, 3)
        self.span_scale_raw = nn.Parameter(torch.tensor(-2.0))
        self.context_scale_raw = nn.Parameter(torch.tensor(-1.0))
        self.intent_scale_raw = nn.Parameter(torch.tensor(-1.0))
        self.span_mix_logits = nn.Parameter(torch.zeros(3)) if self.robust_span else None
        if self.context_gate:
            gate_dim = max(16, units // 4)
            self.context_gate_head = nn.Sequential(
                RMSNorm(context_dim),
                nn.Linear(context_dim, gate_dim),
                nn.SiLU(),
                nn.Linear(gate_dim, 1),
            )
        else:
            self.context_gate_head = None
        self.intent_head = nn.Linear(context_dim, 3) if self.intent_aux else None

    def forward(self, input_ids, word_ids=None):
        byte_mask = input_ids.ne(PAD_ID)
        embedded = self.embedding(input_ids)
        stem_hidden, mask = self.stem(embedded, byte_mask)
        hidden = self.edrn(stem_hidden, mask)
        if self.residual_fusion:
            residual = self.residual_proj(stem_hidden)
            hidden = self.residual_norm(hidden + torch.sigmoid(self.residual_gate) * residual)
            hidden = hidden * mask.unsqueeze(-1)
        pooled = self.pool(hidden, mask)
        if self.word_encoder is not None:
            if word_ids is None:
                raise ValueError("word_ids are required when semantic_context is enabled")
            context = torch.cat([pooled, self.word_encoder(word_ids)], dim=-1)
        else:
            context = pooled
        span_logits = self.span_head(hidden).squeeze(-1)
        masked_span = span_logits.masked_fill(~mask, -1e4)
        span_max = masked_span.max(dim=1).values
        if self.robust_span:
            topk = min(4, masked_span.size(1))
            top_values = masked_span.topk(topk, dim=1).values
            top_valid = top_values.gt(-1e3)
            span_topk = (top_values * top_valid).sum(dim=1) / top_valid.sum(dim=1).clamp_min(1)
            valid_count = mask.sum(dim=1).clamp_min(1)
            span_lse = torch.logsumexp(masked_span, dim=1) - valid_count.log()
            span_sources = torch.stack([span_max, span_topk, span_lse], dim=-1)
            span_evidence = (span_sources * torch.softmax(self.span_mix_logits, dim=0)).sum(dim=-1)
        else:
            span_evidence = span_max
        class_logits = self.classifier(pooled).squeeze(-1)
        gate_logits = self.context_gate_head(context).squeeze(-1) if self.context_gate else None
        intent_logits = self.intent_head(context) if self.intent_head is not None else None
        intent_log_odds = None
        if intent_logits is not None:
            nonattack_logit = torch.logsumexp(intent_logits[:, :2], dim=1)
            intent_log_odds = intent_logits[:, 2] - nonattack_logit
        if self.gate_fusion == "intent_only":
            class_logits = intent_log_odds
        if self.use_span_evidence:
            if gate_logits is not None:
                evidence_gate = torch.sigmoid(gate_logits)
            elif self.gate_fusion == "intent_gated_span":
                evidence_gate = torch.softmax(intent_logits, dim=1)[:, 2]
            else:
                evidence_gate = 1.0
            class_logits = class_logits + evidence_gate * F.softplus(self.span_scale_raw) * span_evidence
        if self.gate_fusion in {"logit_residual", "dual"}:
            class_logits = class_logits + F.softplus(self.context_scale_raw) * gate_logits
        if self.gate_fusion in {"intent_log_odds", "intent_gated_span", "dual"}:
            class_logits = class_logits + F.softplus(self.intent_scale_raw) * intent_log_odds
        return {
            "logits": class_logits,
            "span_logits": span_logits,
            "duration_logits": self.duration_head(pooled),
            "gate_logits": gate_logits,
            "intent_logits": intent_logits,
            "mask": mask,
        }


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon):
        for name, parameter in self.model.named_parameters():
            if "embedding.weight" not in name or parameter.grad is None:
                continue
            norm = parameter.grad.norm()
            if torch.isfinite(norm) and norm.item() > 0:
                self.backup[name] = parameter.data.clone()
                parameter.data.add_(epsilon * parameter.grad / norm)

    def restore(self):
        for name, parameter in self.model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup.clear()


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def inference_params(model):
    duration = sum(parameter.numel() for parameter in model.duration_head.parameters())
    intent = (
        sum(parameter.numel() for parameter in model.intent_head.parameters())
        if model.intent_head is not None and not model.intent_inference
        else 0
    )
    return count_params(model) - duration - intent


def downsample_targets(span_targets, byte_mask, stride):
    span = F.max_pool1d(
        span_targets.unsqueeze(1), kernel_size=stride, stride=stride, ceil_mode=True
    ).squeeze(1)
    mask = F.max_pool1d(
        byte_mask.float().unsqueeze(1), kernel_size=stride, stride=stride, ceil_mode=True
    ).squeeze(1).bool()
    return span, mask


def compute_loss(outputs, batch, args, class_pos_weight):
    input_ids, labels, span_targets, duration_ids, sample_weights, intent_ids, _ = batch
    class_loss = F.binary_cross_entropy_with_logits(
        outputs["logits"],
        labels,
        pos_weight=class_pos_weight,
        reduction="none",
    )
    class_loss = (class_loss * sample_weights).mean()

    down_span, down_mask = downsample_targets(span_targets, input_ids.ne(PAD_ID), args.downsample)
    span_loss = F.binary_cross_entropy_with_logits(
        outputs["span_logits"],
        down_span,
        pos_weight=outputs["span_logits"].new_tensor(args.span_pos_weight),
        reduction="none",
    )
    span_loss = (span_loss * down_mask).sum(dim=1) / down_mask.sum(dim=1).clamp_min(1)
    span_loss = (span_loss * sample_weights).mean()

    duration_valid = duration_ids.ge(0)
    if duration_valid.any():
        duration_loss = F.cross_entropy(
            outputs["duration_logits"][duration_valid], duration_ids[duration_valid]
        )
    else:
        duration_loss = outputs["logits"].new_zeros(())
    if outputs["intent_logits"] is not None:
        intent_loss = F.cross_entropy(outputs["intent_logits"], intent_ids)
    else:
        intent_loss = outputs["logits"].new_zeros(())
    if outputs["gate_logits"] is not None:
        gate_targets = intent_ids.eq(2).float()
        gate_loss = F.binary_cross_entropy_with_logits(
            outputs["gate_logits"], gate_targets, reduction="none"
        )
        gate_loss = (gate_loss * sample_weights).mean()
    else:
        gate_loss = outputs["logits"].new_zeros(())
    total = (
        class_loss
        + args.span_loss_weight * span_loss
        + args.duration_loss_weight * duration_loss
        + args.intent_loss_weight * intent_loss
        + args.gate_loss_weight * gate_loss
    )
    return tuple(
        value.detach() if index else value
        for index, value in enumerate(
            [total, class_loss, span_loss, duration_loss, intent_loss, gate_loss]
        )
    )


def move_batch(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def class_pos_weight(dataset, device):
    labels = dataset.tensors[1]
    positives = labels.sum().item()
    negatives = len(labels) - positives
    value = negatives / positives if positives > 0 else 1.0
    return torch.tensor(value, dtype=torch.float32, device=device)


def optimal_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def constrained_threshold(labels, probabilities, notinject_probs, overdefense_probs, args):
    candidates = np.unique(
        np.concatenate(
            [np.asarray(probabilities), np.asarray(notinject_probs), np.asarray(overdefense_probs), [0.0, 1.0]]
        )
    )
    best = None
    for threshold in candidates:
        metrics = calculate_metrics(labels, probabilities, float(threshold))
        notinject_fpr = float(np.mean(np.asarray(notinject_probs) >= threshold))
        overdefense_fpr = float(np.mean(np.asarray(overdefense_probs) >= threshold))
        violation = max(0.0, notinject_fpr - args.max_fpr_notinject) + max(
            0.0, overdefense_fpr - args.max_fpr_overdefense
        )
        feasible = violation <= 1e-12
        score = metrics["F1"] - 2.0 * violation
        rank = (int(feasible), score, metrics["F1"], -float(threshold))
        if best is None or rank > best[0]:
            best = (rank, float(threshold), metrics, notinject_fpr, overdefense_fpr, feasible, score)
    _, threshold, metrics, notinject_fpr, overdefense_fpr, feasible, score = best
    return threshold, metrics, notinject_fpr, overdefense_fpr, feasible, score


def calculate_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {
        "N": int(len(labels)),
        "PositiveRate": float(labels.mean()) if len(labels) else float("nan"),
        "P": precision_score(labels, predictions, zero_division=0),
        "R": recall_score(labels, predictions, zero_division=0),
        "Acc": accuracy_score(labels, predictions),
        "F1": f1_score(labels, predictions, zero_division=0),
        "FPR": fp / (fp + tn) if fp + tn else 0.0,
        "FNR": fn / (fn + tp) if fn + tp else 0.0,
        "MCC": matthews_corrcoef(labels, predictions) if len(np.unique(labels)) > 1 else float("nan"),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
    if len(np.unique(labels)) > 1:
        result["PRAUC"] = average_precision_score(labels, probabilities)
        result["ROCAUC"] = roc_auc_score(labels, probabilities)
    else:
        result["PRAUC"] = float("nan")
        result["ROCAUC"] = float("nan")
    return result


@torch.no_grad()
def collect_probabilities(model, loader, device, amp_enabled):
    model.eval()
    probabilities, labels = [], []
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(batch[0], batch[6])
            loss = F.binary_cross_entropy_with_logits(outputs["logits"], batch[1])
        probabilities.extend(torch.sigmoid(outputs["logits"]).float().cpu().tolist())
        labels.extend(batch[1].int().cpu().tolist())
        losses.append(float(loss.item()))
    return np.asarray(labels), np.asarray(probabilities), float(np.mean(losses)) if losses else 0.0


def make_scheduler(optimizer, total_steps, warmup_ratio, min_lr_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def schedule(step):
        if step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def append_csv(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_run_keys(save_dir, mode):
    path = save_dir / f"{mode}_runs.csv"
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    required = {"Model", "Seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Cannot resume malformed run file: {path}")
    return set(zip(frame["Model"].astype(str), frame["Seed"].astype(int)))


def train_one(config, datasets, args, seed, device, run_id):
    set_seed(seed)
    model = CompactSelectiveSpanEDRN(config, args.downsample).to(device)
    total_params = count_params(model)
    infer_params = inference_params(model)
    if total_params > args.max_params:
        print(f"[SKIP] {config['name']}: {total_params:,} params exceeds {args.max_params:,}")
        return None

    train_dataset = datasets["train"]
    if config.get("use_context_data", False) and "train_context" in datasets:
        train_dataset = concat_tensor_datasets(train_dataset, datasets["train_context"])
    train_loader = make_loader(train_dataset, args.batch_size, True, seed, args.num_workers)
    val_loader = make_loader(datasets["val"], args.batch_size, False, seed, args.num_workers)
    notinject_loader = make_loader(
        datasets["val_notinject"], args.batch_size, False, seed, args.num_workers
    )
    overdefense_loader = make_loader(
        datasets["val_overdefense"], args.batch_size, False, seed, args.num_workers
    )
    pair_dataset = datasets.get("train_pairs") if config.get("use_pair_loss", False) else None
    pair_loader = (
        make_loader(pair_dataset, args.pair_batch_size, True, seed + 997, args.num_workers)
        if pair_dataset is not None and len(pair_dataset)
        else None
    )
    pair_iterator = iter(pair_loader) if pair_loader is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio, args.min_lr_ratio)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    fgm = FGM(model)
    positive_weight = class_pos_weight(train_dataset, device)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_notinject_fpr = 1.0
    best_overdefense_fpr = 1.0
    best_feasible = False
    best_selection_score = -float("inf")
    bad_epochs = 0
    history_path = args.save_dir / f"{args.mode}_epochs.csv"

    print(
        f"[START] {config['name']} seed={seed} train_params={total_params:,} "
        f"inference_params={infer_params:,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        component_sums = np.zeros(6, dtype=float)
        progress = tqdm(train_loader, desc=f"{config['name']} S{seed} E{epoch}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(batch[0], batch[6])
                loss, cls_loss, span_loss, duration_loss, intent_loss, gate_loss = compute_loss(
                    outputs, batch, args, positive_weight
                )
                pair_loss = outputs["logits"].new_zeros(())
                if pair_iterator is not None and (len(epoch_losses) % max(1, args.pair_every) == 0):
                    try:
                        pair_batch = next(pair_iterator)
                    except StopIteration:
                        pair_iterator = iter(pair_loader)
                        pair_batch = next(pair_iterator)
                    pair_batch = move_batch(pair_batch, device)
                    positive_logits = model(pair_batch[0], pair_batch[1])["logits"]
                    negative_logits = model(pair_batch[2], pair_batch[3])["logits"]
                    pair_loss = F.relu(args.pair_margin - positive_logits + negative_logits).mean()
                    pair_weight = float(config.get("pair_loss_weight", args.pair_loss_weight))
                    loss = loss + pair_weight * pair_loss
            scaler.scale(loss).backward()

            if args.fgm:
                fgm.attack(args.fgm_epsilon)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    adv_outputs = model(batch[0], batch[6])
                    adv_loss, *_ = compute_loss(adv_outputs, batch, args, positive_weight)
                scaler.scale(adv_loss).backward()
                fgm.restore()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_losses.append(float(loss.item()))
            component_sums += [
                float(cls_loss),
                float(span_loss),
                float(duration_loss),
                float(intent_loss),
                float(gate_loss),
                float(pair_loss.detach()),
            ]

        val_labels, val_probs, val_loss = collect_probabilities(model, val_loader, device, amp_enabled)
        _, notinject_probs, _ = collect_probabilities(model, notinject_loader, device, amp_enabled)
        _, overdefense_probs, _ = collect_probabilities(model, overdefense_loader, device, amp_enabled)
        (
            threshold,
            val_metrics,
            notinject_fpr,
            overdefense_fpr,
            feasible,
            selection_score,
        ) = constrained_threshold(val_labels, val_probs, notinject_probs, overdefense_probs, args)
        epoch_row = {
            "RunID": run_id,
            "Mode": args.mode,
            "Model": config["name"],
            "Seed": seed,
            "Epoch": epoch,
            "TrainParams": total_params,
            "InferenceParams": infer_params,
            "TrainLoss": float(np.mean(epoch_losses)),
            "ClassLoss": component_sums[0] / max(1, len(train_loader)),
            "SpanLoss": component_sums[1] / max(1, len(train_loader)),
            "DurationLoss": component_sums[2] / max(1, len(train_loader)),
            "IntentLoss": component_sums[3] / max(1, len(train_loader)),
            "GateLoss": component_sums[4] / max(1, len(train_loader)),
            "PairLoss": component_sums[5] / max(1, len(train_loader)),
            "ValLoss": val_loss,
            "Threshold": threshold,
            "ValNotInjectFPR": notinject_fpr,
            "ValOverdefenseFPR": overdefense_fpr,
            "ThresholdFeasible": feasible,
            "SelectionScore": selection_score,
            **{f"Val{key}": value for key, value in val_metrics.items()},
        }
        append_csv(history_path, epoch_row)

        if selection_score > best_selection_score + 1e-6:
            best_val = val_metrics["F1"]
            best_selection_score = selection_score
            best_epoch = epoch
            best_threshold = threshold
            best_notinject_fpr = notinject_fpr
            best_overdefense_fpr = overdefense_fpr
            best_feasible = feasible
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"[{config['name']} S{seed} E{epoch}] val_f1={val_metrics['F1']:.4f} "
            f"pr_auc={val_metrics['PRAUC']:.4f} ni_fpr={notinject_fpr:.4f} "
            f"od_fpr={overdefense_fpr:.4f} feasible={feasible} th={threshold:.4f}"
        )
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[EARLY STOP] best_epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    row = {
        "RunID": run_id,
        "Mode": args.mode,
        "Model": config["name"],
        "Seed": seed,
        "TrainParams": total_params,
        "InferenceParams": infer_params,
        "SizeMB_FP32": infer_params * 4 / 1024**2,
        "BestEpoch": best_epoch,
        "Threshold": best_threshold,
        "ValF1": best_val,
        "ValNotInjectFPR": best_notinject_fpr,
        "ValOverdefenseFPR": best_overdefense_fpr,
        "ThresholdFeasible": best_feasible,
        "SelectionScore": best_selection_score,
        "ConfigJSON": json.dumps(config, ensure_ascii=True, sort_keys=True),
    }
    val_labels, val_probs, val_loss = collect_probabilities(model, val_loader, device, amp_enabled)
    row.update({f"Val{key}": value for key, value in calculate_metrics(val_labels, val_probs, best_threshold).items()})
    row["ValLoss"] = val_loss

    if args.mode == "final":
        evaluate_final_splits(model, datasets, args, seed, device, run_id, config["name"], best_threshold)
        checkpoint_dir = args.save_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": "ContextGatedEDRN",
                "config": config,
                "downsample": args.downsample,
                "max_bytes": args.max_bytes,
                "threshold": best_threshold,
                "seed": seed,
                "state_dict": best_state,
            },
            checkpoint_dir / f"{config['name']}_seed{seed}.pth",
        )

    result_path = args.save_dir / f"{args.mode}_runs.csv"
    append_csv(result_path, row)
    del model, optimizer, scheduler, scaler, fgm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


@torch.no_grad()
def measure_latency(model, dataset, device, repeats):
    model.eval()
    sample = dataset.tensors[0][:1].to(device)
    sample_words = dataset.tensors[6][:1].to(device)
    for _ in range(10):
        model(sample, sample_words)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        model(sample, sample_words)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / max(1, repeats)


def evaluate_final_splits(model, datasets, args, seed, device, run_id, model_name, threshold):
    amp_enabled = bool(args.amp and device.type == "cuda")
    requested = [name.strip() for name in args.eval_splits.split(",") if name.strip()]
    for split in requested:
        if split not in datasets:
            print(f"[WARN] Evaluation split '{split}' is absent")
            continue
        loader = make_loader(datasets[split], args.batch_size, False, seed, args.num_workers)
        labels, probs, loss = collect_probabilities(model, loader, device, amp_enabled)
        metrics = calculate_metrics(labels, probs, threshold)
        row = {
            "RunID": run_id,
            "Model": model_name,
            "Seed": seed,
            "Split": split,
            "ThresholdFromVal": threshold,
            "Loss": loss,
            "LatencyMsBatch1": float("nan"),
            **metrics,
        }
        if split == "test":
            row["LatencyMsBatch1"] = measure_latency(
                model, datasets[split], device, args.latency_repeats
            )
        if split == "diagnostic_test":
            row["LatencyMsBatch1"] = measure_latency(
                model, datasets[split], device, args.latency_repeats
            )
        append_csv(args.save_dir / "final_slice_metrics.csv", row)
        print(
            f"[FINAL] {model_name} seed={seed} split={split} "
            f"f1={metrics['F1']:.4f} fpr={metrics['FPR']:.4f} fnr={metrics['FNR']:.4f}"
        )


def write_search_summary(save_dir, run_id):
    path = save_dir / "search_runs.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame = frame.drop_duplicates(["Model", "Seed"], keep="last")
    if frame.empty:
        return
    summary = (
        frame.groupby("Model", as_index=False)
        .agg(
            InferenceParams=("InferenceParams", "first"),
            SizeMB_FP32=("SizeMB_FP32", "first"),
            Seeds=("Seed", "nunique"),
            ValF1Mean=("ValF1", "mean"),
            ValF1Std=("ValF1", "std"),
            ValPRAUCMean=("ValPRAUC", "mean"),
            ValFPRMean=("ValFPR", "mean"),
            ValFNRMean=("ValFNR", "mean"),
            ValNotInjectFPRMean=("ValNotInjectFPR", "mean"),
            ValOverdefenseFPRMean=("ValOverdefenseFPR", "mean"),
            FeasibleRate=("ThresholdFeasible", "mean"),
            SelectionScoreMean=("SelectionScore", "mean"),
        )
        .sort_values(["FeasibleRate", "ValF1Mean", "InferenceParams"], ascending=[False, False, True])
    )
    summary["ValF1Std"] = summary["ValF1Std"].fillna(0.0)
    summary["F1Per100KParams"] = summary["ValF1Mean"] / (summary["InferenceParams"] / 100000)
    summary.to_csv(save_dir / "search_summary.csv", index=False)

    pareto_rows = []
    best_f1 = -1.0
    for row in summary.sort_values("InferenceParams").itertuples(index=False):
        if row.ValF1Mean > best_f1 + 1e-9:
            pareto_rows.append(row._asdict())
            best_f1 = row.ValF1Mean
    pd.DataFrame(pareto_rows).to_csv(save_dir / "pareto_front.csv", index=False)


def write_final_summary(save_dir, run_id):
    path = save_dir / "final_slice_metrics.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame = frame.drop_duplicates(["Model", "Seed", "Split"], keep="last")
    if frame.empty:
        return
    summary = (
        frame.groupby(["Model", "Split"], as_index=False)
        .agg(
            Seeds=("Seed", "nunique"),
            F1Mean=("F1", "mean"),
            F1Std=("F1", "std"),
            AccMean=("Acc", "mean"),
            FPRMean=("FPR", "mean"),
            FNRMean=("FNR", "mean"),
            PRAUCMean=("PRAUC", "mean"),
            LatencyMsBatch1=("LatencyMsBatch1", "mean"),
        )
        .sort_values(["Split", "F1Mean"], ascending=[True, False])
    )
    summary.to_csv(save_dir / "final_summary.csv", index=False)


def main():
    args = parse_args()
    if args.max_bytes % args.downsample != 0:
        print("[WARN] max-bytes is not divisible by downsample; ceil pooling will be used")
    if args.mode == "smoke":
        args.epochs = min(args.epochs, 2)
        args.patience = 0
        args.seeds = args.seeds.split(",")[0]
        if args.configs == "dual_resolution":
            args.configs = V7_CONFIGS[0]["name"]
        elif args.configs == "context_v3":
            args.configs = V6_CONFIGS[0]["name"]
        elif args.configs == "context_v2":
            args.configs = V5_CONFIGS[0]["name"]
        elif args.configs == "context_ablation":
            args.configs = CONTEXT_CONFIGS[-1]["name"]
        elif args.configs == "advanced":
            args.configs = ADVANCED_CONFIGS[0]["name"]
        elif args.configs == "improved":
            args.configs = IMPROVED_CONFIGS[0]["name"]
        elif args.configs == "preset":
            args.configs = PRESET_CONFIGS[0]["name"]
        else:
            args.configs = args.configs.split(",")[0].strip()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    device = get_device(args.device)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    configs = load_configs(args.configs)
    if args.mode == "final" and args.configs in {
        "preset", "improved", "advanced", "context_ablation", "context_v2", "context_v3", "dual_resolution"
    }:
        raise ValueError(
            "Final mode requires one explicit frozen config, for example --configs LEDRN_XS_SharedBi"
        )
    frame = load_frame(args.data_file)

    if args.mode == "smoke":
        smoke_parts = []
        for split, limit in [
            ("train", 512),
            ("train_context", 128),
            ("val", 256),
            ("val_notinject", 90),
            ("val_overdefense", 128),
        ]:
            split_frame = frame[frame["split"].eq(split)]
            part = split_frame.groupby("label", group_keys=False).head(
                limit // 2 if split_frame["label"].nunique() > 1 else limit
            )
            smoke_parts.append(part)
        frame = pd.concat(smoke_parts, ignore_index=True)

    print(f"Device: {device}")
    print(f"Data: {args.data_file}")
    print(f"Rows by split: {frame['split'].value_counts().to_dict()}")
    if any(int(config.get("word_vocab", args.word_vocab)) != args.word_vocab for config in configs):
        raise ValueError("All selected configs must use the same --word-vocab as the encoder")
    datasets = encode_all_splits(frame, args.max_bytes, args.max_words, args.word_vocab)
    pair_frame = frame[frame["split"].isin(["train", "train_context"])]
    datasets["train_pairs"] = encode_pair_dataset(
        pair_frame, args.max_bytes, args.max_words, args.word_vocab
    )
    run_config = {
        "run_id": run_id,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "configs": configs,
        "device": str(device),
        "torch": torch.__version__,
    }
    (args.save_dir / f"run_{run_id}.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    completed = (
        completed_run_keys(args.save_dir, args.mode)
        if args.mode in {"search", "final"} and args.resume
        else set()
    )
    for seed in seeds:
        for config in configs:
            key = (config["name"], seed)
            if key in completed:
                print(f"[RESUME SKIP] {config['name']} seed={seed}")
                continue
            train_one(config, datasets, args, seed, device, run_id)
            if args.mode == "search":
                write_search_summary(args.save_dir, run_id)

    if args.mode == "search":
        write_search_summary(args.save_dir, run_id)
        print(f"Search complete. Select configs from: {args.save_dir / 'pareto_front.csv'}")
        print("Do not report test metrics until the selected config is rerun with --mode final.")
    elif args.mode == "final":
        write_final_summary(args.save_dir, run_id)
        print(f"Final evaluation complete: {args.save_dir / 'final_summary.csv'}")
    else:
        print("Smoke test complete")


if __name__ == "__main__":
    main()
