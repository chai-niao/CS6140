import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

from model import N_DURATION_BINS, DualHeadModel, load_model, save_model  # noqa: E402

VOCAB = 100


@pytest.fixture(params=["xlmr", "qwen"])
def tiny(request):
    torch.manual_seed(0)
    return DualHeadModel(request.param, tiny_vocab=VOCAB).eval()


def batch(n=2, length=12):
    torch.manual_seed(1)
    ids = torch.randint(3, VOCAB, (n, length))
    return ids, torch.ones(n, length, dtype=torch.long)


def test_two_heads_with_the_right_shapes(tiny):
    ids, mask = batch()
    sw, dur = tiny(ids, mask)
    assert sw.shape == (2, 12, 2)
    assert dur.shape == (2, 12, N_DURATION_BINS)


def test_prefix_positions_do_not_see_the_future(tiny):
    # changing tokens 6.. must leave the outputs at positions 0..5 untouched (causal constraint)
    ids, mask = batch(1)
    sw1, dur1 = tiny(ids, mask)
    ids2 = ids.clone()
    ids2[0, 6:] = (ids[0, 6:] + 1 - 3) % (VOCAB - 3) + 3  # every future token differs
    sw2, dur2 = tiny(ids2, mask)
    assert torch.allclose(sw1[0, :6], sw2[0, :6], atol=1e-5)
    assert torch.allclose(dur1[0, :6], dur2[0, :6], atol=1e-5)
    assert not torch.allclose(sw1[0, 6:], sw2[0, 6:], atol=1e-5)  # the future itself did change


def test_right_padding_does_not_change_real_positions(tiny):
    ids, mask = batch(1, 8)
    sw_short, _ = tiny(ids, mask)
    pad = torch.full((1, 4), 1 if tiny.is_encoder_type else 0, dtype=torch.long)
    sw_padded, _ = tiny(torch.cat([ids, pad], 1), torch.cat([mask, torch.zeros(1, 4, dtype=torch.long)], 1))
    assert torch.allclose(sw_short[0], sw_padded[0, :8], atol=1e-4)


def test_train_top_layers_only_freezes_the_rest(tiny):
    tiny.train_top_layers_only(1)
    layers = tiny.transformer_layers()
    assert all(not p.requires_grad for p in layers[0].parameters())
    assert all(p.requires_grad for p in layers[-1].parameters())
    assert all(p.requires_grad for p in tiny.switch_head.parameters())
    assert all(p.requires_grad for p in tiny.duration_head.parameters())
    emb = tiny.backbone.get_input_embeddings()
    assert not emb.weight.requires_grad
    total, trainable = tiny.count_parameters()
    assert 0 < trainable < total


def test_checkpoint_roundtrip(tiny, tmp_path):
    ids, mask = batch()
    sw1, dur1 = tiny(ids, mask)
    save_model(tiny, tmp_path / "m.pt", {"step": 3})
    again = load_model(tmp_path / "m.pt", tiny_vocab=VOCAB).eval()
    sw2, dur2 = again(ids, mask)
    assert torch.allclose(sw1, sw2, atol=2e-2)  # fp16 checkpoint
    assert torch.allclose(dur1, dur2, atol=2e-2)


# ---------------------------------------------------------------- attention ablation (train without the mask, test with it)
def future_changes_prefix(model) -> bool:
    """True if replacing tokens 6.. changes the switch logits at positions 0..5."""
    ids, mask = batch(1)
    sw1, _ = model(ids, mask)
    ids2 = ids.clone()
    ids2[0, 6:] = (ids[0, 6:] + 1 - 3) % (VOCAB - 3) + 3
    sw2, _ = model(ids2, mask)
    return not torch.allclose(sw1[0, :6], sw2[0, :6], atol=1e-5)


def test_bidirectional_encoder_sees_the_future_and_set_attention_removes_it():
    torch.manual_seed(0)
    m = DualHeadModel("xlmr", tiny_vocab=VOCAB, attention="bidirectional").eval()
    assert future_changes_prefix(m)
    m.set_attention("causal")
    assert not future_changes_prefix(m)
    m.set_attention("bidirectional")
    assert future_changes_prefix(m)


def test_causal_and_switched_models_agree():
    # a bidirectional model switched to causal must give exactly what a causal model with the same weights gives
    torch.manual_seed(0)
    bi = DualHeadModel("xlmr", tiny_vocab=VOCAB, attention="bidirectional").eval()
    causal = DualHeadModel("xlmr", tiny_vocab=VOCAB, attention="causal").eval()
    causal.load_state_dict(bi.state_dict())
    bi.set_attention("causal")
    ids, mask = batch()
    assert torch.allclose(bi(ids, mask)[0], causal(ids, mask)[0], atol=1e-6)


def test_decoder_has_no_bidirectional_mode():
    with pytest.raises(ValueError):
        DualHeadModel("qwen", tiny_vocab=VOCAB, attention="bidirectional")
    with pytest.raises(ValueError):
        DualHeadModel("qwen", tiny_vocab=VOCAB).set_attention("bidirectional")


def test_checkpoint_keeps_the_training_attention(tmp_path):
    torch.manual_seed(0)
    m = DualHeadModel("xlmr", tiny_vocab=VOCAB, attention="bidirectional").eval()
    save_model(m, tmp_path / "m.pt")
    assert load_model(tmp_path / "m.pt", tiny_vocab=VOCAB).attention == "bidirectional"
    assert not future_changes_prefix(load_model(tmp_path / "m.pt", tiny_vocab=VOCAB, attention="causal").eval())
