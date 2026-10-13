"""Step 5: enforce and verify the prefix-only (causal) constraint.

Model A  xlm-roberta-base is bidirectional by default. Setting `config.is_decoder = True`
         makes Hugging Face apply a causal (look-ahead) attention mask, so position t can
         only attend to positions <= t. No weights change.
Model B  Qwen2.5-0.5B is a decoder-only Transformer and is causal by construction.

Leak test 1 (one sentence): encode a sentence, then replace every token AFTER position t
with random tokens. If the hidden states at positions <= t do not change, nothing from the
future leaks into the prefix. We also run XLM-R WITHOUT the mask to show what leakage
looks like.

Leak test 2 (real batch): the same check on a batch produced by the training DataLoader
(dataset.make_loader: BOS token, word-by-word subwords, right padding, attention mask),
with conversations from several pairs of the dev split and a random labelled position t
in every row. It also checks that padding does not change the hidden states: each row
run alone must match the same row inside the padded batch.

Only forward passes on CPU; the model weights are downloaded once (about 1 GB each).
Writes results/leak_test.txt.
"""
from __future__ import annotations

import argparse
import random
import time

import torch

from common import RESULTS_DIR, load_split
from dataset import bos_id, encode_words, make_loader

SENTENCE = "Cuando termine mi maestría, I will invest in sustainable projects.".split()
T_WORD = 3  # index of "maestría," : the position where a switch is predicted


def load(name: str, causal: bool):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    cfg = AutoConfig.from_pretrained(name)
    if "roberta" in name.lower():
        cfg.is_decoder = causal  # the switch that turns on the causal mask in RobertaModel
    model = AutoModel.from_pretrained(name, config=cfg, attn_implementation="eager")
    model.eval()
    return tok, model


@torch.no_grad()
def hidden_states(model, ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    return out.last_hidden_state[0]


def leak_test(tok, model, words: list[str], t_word: int, trials: int = 3, seed: int = 0) -> dict:
    ids, wids = encode_words(words, [True] * len(words), tok)
    if bos_id(tok) is not None:  # same input layout as the DataLoader
        ids, wids = [bos_id(tok)] + ids, [None] + wids
    ids = torch.tensor([ids])
    t = max(i for i, w in enumerate(wids) if w == t_word)  # last subword of the target word
    n = ids.shape[1]
    base = hidden_states(model, ids)
    vocab = min(len(tok), model.get_input_embeddings().num_embeddings)
    rng = random.Random(seed)
    prefix_diffs, suffix_diffs = [], []
    for _ in range(trials):
        corrupt = ids.clone()
        for j in range(t + 1, n):
            corrupt[0, j] = rng.randrange(vocab)
        h = hidden_states(model, corrupt)
        prefix_diffs.append((h[: t + 1] - base[: t + 1]).abs().max().item())
        suffix_diffs.append((h[t + 1 :] - base[t + 1 :]).abs().max().item())
    return {"t": t, "n": n, "prefix_max_diff": max(prefix_diffs), "suffix_max_diff": max(suffix_diffs)}


BATCH_PAIRS = ["Spanish_eng", "Chinese_eng", "Japanese_eng", "Hindi_eng"]


def real_batch(tok, per_pair: int = 2) -> dict:
    """First batch of the training DataLoader over a few dev conversations of several pairs."""
    convs = []
    for pair in BATCH_PAIRS:
        convs += load_split(pair, "dev")[:per_pair]
    loader = make_loader(convs, tok, batch_size=len(convs), max_len=128, stride=64, shuffle=False)
    return next(iter(loader))


@torch.no_grad()
def leak_test_batch(tok, model, batch: dict, seed: int = 0) -> dict:
    ids, mask = batch["input_ids"], batch["attention_mask"]
    base = model(input_ids=ids, attention_mask=mask).last_hidden_state
    rng = random.Random(seed)
    vocab = min(len(tok), model.get_input_embeddings().num_embeddings)
    corrupt = ids.clone()
    cut = []
    for r in range(ids.shape[0]):
        n = int(mask[r].sum())
        labelled = [i for i in range(n - 1) if batch["y_sw"][r, i] >= 0]
        t = rng.choice(labelled)  # a real prediction position
        cut.append(t)
        for j in range(t + 1, n):
            corrupt[r, j] = rng.randrange(vocab)
    h = model(input_ids=corrupt, attention_mask=mask).last_hidden_state
    prefix_diff = max((h[r, : t + 1] - base[r, : t + 1]).abs().max().item() for r, t in enumerate(cut))
    # padding check: every row alone, without padding, against the same row in the batch
    pad_diff = 0.0
    for r in range(ids.shape[0]):
        n = int(mask[r].sum())
        alone = model(input_ids=ids[r : r + 1, :n], attention_mask=mask[r : r + 1, :n]).last_hidden_state[0]
        pad_diff = max(pad_diff, (alone - base[r, :n]).abs().max().item())
    lengths = mask.sum(1).tolist()
    return {"rows": ids.shape[0], "min_len": min(lengths), "max_len": max(lengths),
            "prefix_max_diff": prefix_diff, "pad_max_diff": pad_diff}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["xlmr_bidirectional", "xlmr_causal", "qwen"])
    args = ap.parse_args()
    configs = {
        "xlmr_bidirectional": ("xlm-roberta-base", False, "XLM-R without mask (reference)"),
        "xlmr_causal": ("xlm-roberta-base", True, "Model A: XLM-R + causal mask"),
        "qwen": ("Qwen/Qwen2.5-0.5B", True, "Model B: Qwen2.5-0.5B (causal by construction)"),
    }
    lines = [f"Leak test. Sentence: {' '.join(SENTENCE)}", f"Target word t = {T_WORD + 1} ({SENTENCE[T_WORD]!r}); all tokens after it are replaced by random ids, 3 trials.", ""]
    lines.append("| model | causal | subwords | t (last subword) | max |diff| at positions <= t | max |diff| after t | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    batch_rows: list[str] = []
    for key in args.models:
        name, causal, label = configs[key]
        t0 = time.time()
        tok, model = load(name, causal)
        r = leak_test(tok, model, SENTENCE, T_WORD)
        verdict = "no leak: prefix unchanged" if r["prefix_max_diff"] < 1e-4 else "LEAK: prefix depends on future tokens"
        lines.append(f"| {label} | {'yes' if causal else 'no'} | {r['n']} | {r['t']} | {r['prefix_max_diff']:.2e} | {r['suffix_max_diff']:.2e} | {verdict} |")
        print(lines[-1], f"({time.time() - t0:.0f}s)")
        rb = leak_test_batch(tok, model, real_batch(tok))
        verdict = "no leak" if rb["prefix_max_diff"] < 1e-4 else "LEAK"
        verdict += ", padding-invariant" if rb["pad_max_diff"] < 1e-4 else ", padding changes the output"
        batch_rows.append(f"| {label} | {rb['rows']} | {rb['min_len']} to {rb['max_len']} | {rb['prefix_max_diff']:.2e} | {rb['pad_max_diff']:.2e} | {verdict} |")
        print(batch_rows[-1])
        del model
    lines += ["", f"Leak test on a real DataLoader batch: {len(BATCH_PAIRS) * 2} dev conversations ({', '.join(BATCH_PAIRS)}), "
              "BOS + right padding; in every row the tokens after a random labelled position t are replaced by random ids.", "",
              "| model | rows | subwords per row | max |diff| at positions <= t | max |diff| row alone vs in padded batch | verdict |",
              "|---|---|---|---|---|---|"] + batch_rows
    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "leak_test.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"wrote {RESULTS_DIR / 'leak_test.txt'}")


if __name__ == "__main__":
    main()
