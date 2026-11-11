"""Weighted multi-task loss.

    L = lambda_switch * CE(switch logits, y_sw) + lambda_duration * CE(duration logits, y_dur)

Both cross-entropies use ignore_index = -1:
  * y_sw = -1 on subwords that are not the last subword of a word, on BOS, on padding and on
    positions that an overlapping window has already scored;
  * y_dur = -1 at all of those and, in addition, wherever there is no switch (y_sw = 0),
    because the burst length of the next language is only defined when the language changes.
Optional class weights counter the 87 / 13 imbalance of the switch labels and the uneven
burst-length bins. Logits are up-cast to fp32 so that the loss is exact under mixed precision.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IGNORE = -1


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """Mean cross-entropy over the positions whose target is not -1 (0 if there is none)."""
    logits = logits.reshape(-1, logits.size(-1)).float()
    targets = targets.reshape(-1)
    if not (targets != IGNORE).any():
        return logits.sum() * 0.0  # keeps the graph alive, avoids NaN from an empty mean
    return F.cross_entropy(logits, targets, weight=weight, ignore_index=IGNORE)


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_switch: float = 1.0, lambda_duration: float = 1.0,
                 switch_weight: list[float] | None = None, duration_weight: list[float] | None = None):
        super().__init__()
        self.lambda_switch = float(lambda_switch)
        self.lambda_duration = float(lambda_duration)
        self.register_buffer("switch_weight", torch.tensor(switch_weight, dtype=torch.float32) if switch_weight else None)
        self.register_buffer("duration_weight", torch.tensor(duration_weight, dtype=torch.float32) if duration_weight else None)

    def forward(self, switch_logits: torch.Tensor, duration_logits: torch.Tensor,
                y_sw: torch.Tensor, y_dur: torch.Tensor) -> dict[str, torch.Tensor]:
        l_sw = masked_cross_entropy(switch_logits, y_sw, self.switch_weight)
        l_dur = masked_cross_entropy(duration_logits, y_dur, self.duration_weight)
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
