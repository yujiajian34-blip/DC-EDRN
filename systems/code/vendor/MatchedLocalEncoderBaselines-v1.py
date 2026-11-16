#!/usr/bin/env python
"""Matched local-encoder replacements for the V7 dual-resolution model.

This experiment keeps the byte stem, word-context branch, pooling, prediction
heads, losses, data splits, and constrained threshold selection fixed.  Only
the local sequence encoder is changed.  It is therefore an architectural
control experiment, not a collection of independently tuned baselines.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
V7_PATH = HERE / "DualResolutionCGEDRN-v7-search.py"


def load_v7_module():
    if not V7_PATH.exists():
        raise FileNotFoundError(
            f"Required V7 implementation was not found: {V7_PATH}. "
            "Copy both scripts into the same directory."
        )
    spec = importlib.util.spec_from_file_location("dual_resolution_edrn_v7", V7_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {V7_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V7 = load_v7_module()
BaseModel = V7.CompactSelectiveSpanEDRN


class MaskedDepthwiseBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.pointwise_in = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = V7.RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        residual = hidden
        encoded = self.depthwise(hidden.transpose(1, 2))
        encoded = F.glu(self.pointwise_in(encoded), dim=1)
        encoded = self.pointwise_out(encoded).transpose(1, 2)
        hidden = self.norm(residual + self.dropout(encoded))
        return hidden * mask.unsqueeze(-1)


class ByteCNNLocalEncoder(nn.Module):
    """A compact dilated depthwise CNN with the same output width as BiEDRN."""

    def __init__(self, input_dim: int, units: int, dropout: float):
        super().__init__()
        self.output_dim = 2 * units
        self.input_proj = nn.Linear(input_dim, units)
        self.blocks = nn.ModuleList(
            [MaskedDepthwiseBlock(units, kernel, dropout) for kernel in (3, 5, 9)]
        )
        self.output_proj = nn.Linear(units, self.output_dim)
        self.output_norm = V7.RMSNorm(self.output_dim)

    def forward(self, hidden, mask):
        hidden = F.gelu(self.input_proj(hidden)) * mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.output_norm(self.output_proj(hidden))
        return hidden * mask.unsqueeze(-1)


class PackedRNNLocalEncoder(nn.Module):
    """Padding-safe bidirectional GRU/LSTM replacement."""

    def __init__(self, kind: str, input_dim: int, units: int, dropout: float):
        super().__init__()
        if kind == "bigru":
            hidden_size = max(32, round(units * 0.60))
            rnn_type = nn.GRU
        elif kind == "bilstm":
            hidden_size = max(28, round(units * 0.50))
            rnn_type = nn.LSTM
        else:
            raise ValueError(f"Unknown recurrent encoder: {kind}")
        self.output_dim = 2 * units
        self.rnn = rnn_type(
            input_dim,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Linear(2 * hidden_size, self.output_dim)
        self.norm = V7.RMSNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        lengths = mask.sum(dim=1).clamp_min(1).to(torch.int64).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=hidden.size(1)
        )
        encoded = self.norm(self.projection(self.dropout(encoded)))
        return encoded * mask.unsqueeze(-1)


class TinyTransformerLocalEncoder(nn.Module):
    """One-layer Transformer control with deterministic sinusoidal positions."""

    def __init__(self, input_dim: int, units: int, dropout: float):
        super().__init__()
        model_dim = input_dim
        heads = 4 if model_dim % 4 == 0 else 2
        feedforward_dim = max(96, round(model_dim * 1.5))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_dim = 2 * units
        self.output_proj = nn.Linear(model_dim, self.output_dim)
        self.norm = V7.RMSNorm(self.output_dim)

    @staticmethod
    def positional_encoding(length: int, dim: int, device, dtype):
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype).unsqueeze(0)

    def forward(self, hidden, mask):
        hidden = hidden + self.positional_encoding(
            hidden.size(1), hidden.size(2), hidden.device, hidden.dtype
        )
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.norm(self.output_proj(hidden))
        return hidden * mask.unsqueeze(-1)


class MatchedLocalEncoderModel(BaseModel):
    def __init__(self, config, downsample):
        super().__init__(config, downsample)
        encoder = config.get("local_encoder", "selective_edrn")
        if encoder == "selective_edrn":
            return

        input_dim = int(config["model_dim"])
        units = int(config["units"])
        dropout = float(config["dropout"])
        if encoder == "bytecnn":
            replacement = ByteCNNLocalEncoder(input_dim, units, dropout)
        elif encoder in {"bigru", "bilstm"}:
            replacement = PackedRNNLocalEncoder(encoder, input_dim, units, dropout)
        elif encoder == "tiny_transformer":
            replacement = TinyTransformerLocalEncoder(input_dim, units, dropout)
        else:
            raise ValueError(f"Unknown local_encoder: {encoder}")

        if replacement.output_dim != self.edrn.output_dim:
            raise RuntimeError(
                f"Replacement width {replacement.output_dim} != EDRN width {self.edrn.output_dim}"
            )
        self.edrn = replacement


def make_config(name: str, local_encoder: str):
    config = dict(V7._V7_MEDIUM)
    config.update(
        {
            "name": name,
            "local_encoder": local_encoder,
            "experiment_family": "matched_local_encoder_v1",
        }
    )
    return config


MATCHED_CONFIGS = [
    make_config("Matched_SelectiveEDRN", "selective_edrn"),
    make_config("Matched_ByteCNN", "bytecnn"),
    make_config("Matched_BiGRU", "bigru"),
    make_config("Matched_BiLSTM", "bilstm"),
    make_config("Matched_TinyTransformer", "tiny_transformer"),
]


def load_matched_configs(value):
    presets = {config["name"]: config for config in MATCHED_CONFIGS}
    if value == "matched_local":
        return [dict(config) for config in MATCHED_CONFIGS]
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(presets))
    if unknown:
        raise ValueError(
            f"Unknown matched configs {unknown}; available: {sorted(presets)} or matched_local"
        )
    return [dict(presets[name]) for name in names]


def main():
    # The CUDA memory-efficient attention kernel is non-deterministic on some
    # PyTorch/RTX 3090 combinations. Use the math kernel for reproducible runs.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    V7.CompactSelectiveSpanEDRN = MatchedLocalEncoderModel
    V7.load_configs = load_matched_configs
    V7.main()


if __name__ == "__main__":
    main()
