"""Algebraically unchanged inference specialization for the existing DC checkpoint.

There is no parallel-prefix or affine-scan approximation here. Both directions
retain their original nonlinear, sequential recurrence and parameter names.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F


class SingleStateDirection(nn.Module):
    def __init__(self, original):
        super().__init__()
        if original.d_max != 1 or original.selective_duration:
            raise ValueError("Optimization requires d_max=1 and selective_duration=False")
        self.units = original.units
        self.d_max = original.d_max
        self.reverse = original.reverse
        self.selective_duration = original.selective_duration
        self.duration_prior = original.duration_prior
        # Reuse the supplied parameter objects, including the inactive one-bin
        # start logits, so state_dict names and parameter counts are preserved.
        self.x_proj = original.x_proj
        self.h_proj = original.h_proj
        self.gate_proj = original.gate_proj
        self.duration_delta = original.duration_delta
        self.stay_delta = original.stay_delta
        self.duration_temperature = original.duration_temperature
        self.start_logits = original.start_logits
        self.stay_logits = original.stay_logits
        self.boundary_transition = original.boundary_transition
        self.out_norm = original.out_norm

    def forward(self, x, mask, reverse=None, projected=None, stay=None):
        batch, seq_len, _ = x.shape
        projected = self.x_proj(x) if projected is None else projected
        stay = torch.sigmoid(self.stay_logits[:, 0]).unsqueeze(0) if stay is None else stay
        # softmax over the sole duration bin is identically one. The shifted
        # duration propagation is empty. No learned values are approximated.
        hidden = x.new_zeros(batch, self.units)
        alpha = x.new_zeros(batch, self.units)
        outputs = [None] * seq_len
        run_reverse = self.reverse if reverse is None else bool(reverse)
        steps = range(seq_len - 1, -1, -1) if run_reverse else range(seq_len)
        for t in steps:
            valid = mask[:, t].view(batch, 1)
            core = F.silu(projected[:, t] + self.h_proj(hidden))
            keep_logit, reset_logit, candidate, output_gate = self.gate_proj(core).chunk(4, dim=-1)
            mixture = torch.softmax(torch.stack([keep_logit, reset_logit], dim=-1), dim=-1)
            propagated = alpha * stay
            boundary = alpha * (1.0 - stay)
            candidate = torch.tanh(candidate + self.boundary_transition(boundary))
            alpha_new = mixture[..., 0] * propagated + mixture[..., 1] * candidate
            hidden_new = self.out_norm(torch.sigmoid(output_gate) * torch.tanh(alpha_new))
            alpha = torch.where(valid, alpha_new, alpha)
            hidden = torch.where(valid, hidden_new, hidden)
            outputs[t] = hidden * valid
        return torch.stack(outputs, dim=1)


class StaticHoistedDirection(SingleStateDirection):
    """Conservative fallback retaining the original per-step arithmetic/shapes."""
    def forward(self, x, mask, reverse=None):
        batch, seq_len, _ = x.shape
        hidden = x.new_zeros(batch, self.units)
        alpha = x.new_zeros(batch, self.units, 1)
        start = torch.softmax(self.start_logits, dim=-1).unsqueeze(0).expand(batch, -1, -1)
        stay = torch.sigmoid(self.stay_logits).unsqueeze(0).expand(batch, -1, -1)
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
            propagated = alpha * stay
            propagated[:, :, 1:] = propagated[:, :, 1:] + alpha[:, :, :-1] * (1.0 - stay[:, :, :-1])
            boundary = alpha[:, :, -1] * (1.0 - stay[:, :, -1])
            candidate = torch.tanh(candidate + self.boundary_transition(boundary))
            alpha_new = keep * propagated + restart * candidate.unsqueeze(-1) * start
            hidden_new = self.out_norm(torch.sigmoid(output_gate) * torch.tanh(alpha_new.sum(dim=-1)))
            alpha = torch.where(valid.unsqueeze(-1), alpha_new, alpha)
            hidden = torch.where(valid, hidden_new, hidden)
            outputs[t] = hidden * valid
        return torch.stack(outputs, dim=1)


class SharedSingleStateBiEDRN(nn.Module):
    def __init__(self, original, optimization="precompute"):
        super().__init__()
        if not original.shared_directions or original.direction_fusion != "concat":
            raise ValueError("Optimization requires the existing shared/concat DC configuration")
        self.shared_directions = original.shared_directions
        self.direction_fusion = original.direction_fusion
        self.output_dim = original.output_dim
        if optimization not in ("precompute", "static_hoist"):
            raise ValueError(f"Unknown optimization: {optimization}")
        self.optimization = optimization
        direction = SingleStateDirection if optimization == "precompute" else StaticHoistedDirection
        self.shared_edrn = direction(original.shared_edrn)
        self.forward_edrn = original.forward_edrn
        self.backward_edrn = original.backward_edrn
        self.direction_logits = original.direction_logits
        self.norm = original.norm

    def forward(self, x, mask):
        if self.optimization == "static_hoist":
            forward = self.shared_edrn(x, mask, reverse=False)
            backward = self.shared_edrn(x, mask, reverse=True)
            return self.norm(torch.cat([forward, backward], dim=-1)) * mask.unsqueeze(-1)
        # Input projections and static persistence are shared between directions.
        projected = self.shared_edrn.x_proj(x)
        stay = torch.sigmoid(self.shared_edrn.stay_logits[:, 0]).unsqueeze(0)
        forward = self.shared_edrn(x, mask, reverse=False, projected=projected, stay=stay)
        backward = self.shared_edrn(x, mask, reverse=True, projected=projected, stay=stay)
        return self.norm(torch.cat([forward, backward], dim=-1)) * mask.unsqueeze(-1)


def optimized_copy(original, optimization="precompute"):
    """Copy an inference model and change only the execution of its DC encoder."""
    result = copy.deepcopy(original)
    result.edrn = SharedSingleStateBiEDRN(result.edrn, optimization=optimization)
    before, after = original.state_dict(), result.state_dict()
    if list(before) != list(after):
        raise AssertionError("Optimization changed checkpoint parameter names")
    for name in before:
        if not torch.equal(before[name], after[name]):
            raise AssertionError(f"Optimization changed parameter {name}")
    return result.eval()
