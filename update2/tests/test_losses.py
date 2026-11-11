import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import setup_path  # noqa: E402, F401
from labels import make_labels  # noqa: E402
from losses import IGNORE, MultiTaskLoss, masked_cross_entropy  # noqa: E402


def test_ignored_positions_are_left_out():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 2)
    y = torch.tensor([[0, 1, IGNORE, IGNORE, 1]])
    loss = masked_cross_entropy(logits, y, None)
    expected = F.cross_entropy(logits[0, [0, 1, 4]], y[0, [0, 1, 4]])
    assert torch.allclose(loss, expected)
    # the logits at ignored positions do not matter at all
    logits2 = logits.clone()
    logits2[0, 2:4] += 100.0
    assert torch.allclose(masked_cross_entropy(logits2, y, None), expected)


def test_all_ignored_gives_zero_and_keeps_the_graph():
    logits = torch.randn(2, 4, 3, requires_grad=True)
    y = torch.full((2, 4), IGNORE)
    loss = masked_cross_entropy(logits, y, None)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert logits.grad is not None


def test_lambda_weights_combine_the_two_losses():
    torch.manual_seed(0)
    sw, dur = torch.randn(2, 6, 2), torch.randn(2, 6, 3)
    y_sw = torch.randint(0, 2, (2, 6))
    y_dur = torch.where(y_sw == 1, torch.randint(0, 3, (2, 6)), torch.full((2, 6), IGNORE))
    out = MultiTaskLoss(2.0, 0.5)(sw, dur, y_sw, y_dur)
    assert torch.allclose(out["total"], 2.0 * out["switch"] + 0.5 * out["duration"])
    assert int(out["n_duration_targets"]) == int((y_sw == 1).sum())


def test_class_weight_raises_the_cost_of_missing_a_switch():
    logits = torch.tensor([[[2.0, 0.0]]])  # confidently predicts "no switch"
    y = torch.tensor([[1]])  # but it was a switch
    plain = MultiTaskLoss(switch_weight=None)(logits, torch.zeros(1, 1, 3), y, torch.tensor([[IGNORE]]))["switch"]
    weighted = MultiTaskLoss(switch_weight=[1.0, 5.0])(logits, torch.zeros(1, 1, 3), y, torch.tensor([[IGNORE]]))["switch"]
    # with one target the weighted mean equals the plain loss; the gradient does not, so compare with two targets
    assert torch.allclose(plain, weighted)
    logits2 = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
    y2 = torch.tensor([[1, 0]])
    plain2 = MultiTaskLoss(switch_weight=None)(logits2, torch.zeros(1, 2, 3), y2, torch.full((1, 2), IGNORE))["switch"]
    weighted2 = MultiTaskLoss(switch_weight=[1.0, 5.0])(logits2, torch.zeros(1, 2, 3), y2, torch.full((1, 2), IGNORE))["switch"]
    assert weighted2 > plain2


def test_duration_targets_exist_only_at_switches():
    langs = ["es"] * 4 + ["en"] * 6 + ["es"] * 2
    y_sw, y_dur = make_labels(langs)
    for s, d in zip(y_sw, y_dur):
        assert (d != IGNORE) == (s == 1)
