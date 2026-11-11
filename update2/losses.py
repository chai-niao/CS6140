"""Weighted multi-task loss.

    L = lambda_switch * CE(switch logits, y_sw) + lambda_duration * CE(duration logits, y_dur)

Both cross-entropies use ignore_index = -1:
  * y_sw = -1 on subwords that are not the last subword of a word, on BOS, on padding and on
    positions that an overlapping window has already scored;
  * y_dur = -1 at all of those and, in addition, wherever there is no switch (y_sw = 0),
    because the burst length of the next language is only defined when the language changes.
Optional class weights counter the 87 / 13 imbalance of the switch labels and the uneven
burst-length bins. Logits are up-cast to fp32 so that the loss is exact under mixed precision.

Gradient accumulation: a mean per micro-batch would give every micro-batch the same weight, however
many labelled positions it holds (and length-bucketed micro-batches differ a lot). The training loop
therefore passes the normalizers of the whole optimizer step (`normalizers()` summed over its
micro-batches), so that the accumulated loss is the weighted mean over all positions of the step,
exactly what one large batch would give.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IGNORE = -1


def weighted_count(targets: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """Denominator of the weighted mean cross-entropy: the sum of the class weights of the targets."""
    targets = targets.reshape(-1)
    keep = targets != IGNORE
    if weight is None:
        return keep.sum().float()
    return weight.to(targets.device)[targets[keep]].sum().float()


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None,
                         norm: torch.Tensor | float | None = None) -> torch.Tensor:
    """Weighted cross-entropy over the positions whose target is not -1, divided by `norm`.

    Without `norm` it is the weighted mean over this batch (as F.cross_entropy). A batch without any
    target gives 0 (a sum over nothing) instead of NaN, with no host sync to check for it."""
    logits = logits.reshape(-1, logits.size(-1)).float()
    targets = targets.reshape(-1)
    total = F.cross_entropy(logits, targets, weight=weight, ignore_index=IGNORE, reduction="sum")
    if norm is None:
        norm = weighted_count(targets, weight)
    return total / torch.as_tensor(norm, dtype=total.dtype, device=total.device).clamp(min=1e-8)


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_switch: float = 1.0, lambda_duration: float = 1.0,
                 switch_weight: list[float] | None = None, duration_weight: list[float] | None = None):
        super().__init__()
        self.lambda_switch = float(lambda_switch)
        self.lambda_duration = float(lambda_duration)
        self.register_buffer("switch_weight", torch.tensor(switch_weight, dtype=torch.float32) if switch_weight else None)
        self.register_buffer("duration_weight", torch.tensor(duration_weight, dtype=torch.float32) if duration_weight else None)

    def normalizers(self, y_sw: torch.Tensor, y_dur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted target counts of one micro-batch; sum them over the micro-batches of an optimizer step."""
        return weighted_count(y_sw, self.switch_weight), weighted_count(y_dur, self.duration_weight)

    def forward(self, switch_logits: torch.Tensor, duration_logits: torch.Tensor,
                y_sw: torch.Tensor, y_dur: torch.Tensor,
                norm_switch: torch.Tensor | None = None, norm_duration: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """With norm_switch / norm_duration (step totals from normalizers()), the terms of all micro-batches of a
        step add up to the weighted mean over the step; without them each term is the mean over this batch."""
        l_sw = masked_cross_entropy(switch_logits, y_sw, self.switch_weight, norm_switch)
        l_dur = masked_cross_entropy(duration_logits, y_dur, self.duration_weight, norm_duration)
        return {
            "total": self.lambda_switch * l_sw + self.lambda_duration * l_dur,
            "switch": l_sw,
            "duration": l_dur,
            "n_switch_targets": (y_sw != IGNORE).sum(),
            "n_duration_targets": (y_dur != IGNORE).sum(),
        }

    def describe(self) -> str:
        sw = self.switch_weight.tolist() if self.switch_weight is not None else "none"
        du = self.duration_weight.tolist() if self.duration_weight is not None else "none"
        return (f"L = {self.lambda_switch:g} * CE_switch + {self.lambda_duration:g} * CE_duration, ignore_index = -1, "
                f"switch class weights {sw}, duration class weights {du}")
