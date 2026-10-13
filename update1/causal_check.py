"""Step 5: enforce and verify the prefix-only (causal) constraint.

Model A  xlm-roberta-base is bidirectional by default. Setting `config.is_decoder = True`
         makes Hugging Face apply a causal (look-ahead) attention mask, so position t can
         only attend to positions <= t. No weights change.
Model B  Qwen2.5-0.5B is a decoder-only Transformer and is causal by construction.

Leak test: encode a sentence, then replace every token AFTER position t with random
tokens. If the hidden states at positions <= t do not change, nothing from the future
leaks into the prefix. We also run XLM-R WITHOUT the mask to show what leakage looks like.

Only forward passes on CPU; the model weights are downloaded once (about 1 GB each).
Writes results/leak_test.txt.
"""
from __future__ import annotations

import argparse
import random
import time

import torch

from common import RESULTS_DIR

SENTENCE = "Cuando termine mi maestría, I will invest in sustainable projects.".split()
T_WORD = 3  # index of "maestría," : the position where a switch is predicted


def load(name: str, causal: bool):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    kwargs = {"add_prefix_space": True} if "qwen" in name.lower() else {}
    tok = AutoTokenizer.from_pretrained(name, **kwargs)
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
    enc = tok(words, is_split_into_words=True, add_special_tokens=False)
    ids = torch.tensor([enc["input_ids"]])
    wids = enc.word_ids()
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
    for key in args.models:
        name, causal, label = configs[key]
        t0 = time.time()
        tok, model = load(name, causal)
        r = leak_test(tok, model, SENTENCE, T_WORD)
        verdict = "no leak: prefix unchanged" if r["prefix_max_diff"] < 1e-4 else "LEAK: prefix depends on future tokens"
        lines.append(f"| {label} | {'yes' if causal else 'no'} | {r['n']} | {r['t']} | {r['prefix_max_diff']:.2e} | {r['suffix_max_diff']:.2e} | {verdict} |")
        print(lines[-1], f"({time.time() - t0:.0f}s)")
        del model
    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "leak_test.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"wrote {RESULTS_DIR / 'leak_test.txt'}")


if __name__ == "__main__":
    main()
