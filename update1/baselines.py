"""Step 3: naive baselines and Anticipatory F1 for every pair (no neural network).

A naive model that predicts a switch from the current token only is the reference point
for the transformer models. Four reference points are reported, all on the test split:

  never     always predict "no switch"            -> F1 = 0 by definition
  always    always predict "switch"               -> F1 = 2p / (1 + p), p = switch rate
  lang      P(switch | language of x_t)           -> a per-language constant
  token     P(switch | x_t)  (the required one)   -> lookup table from the training split,
            with additive smoothing toward the pair's prior; the decision threshold is
            tuned on the dev split for F1, then frozen for the test split.

For duration we report accuracy at true switches of (a) the majority bin from training
and (b) the most frequent bin after the current token.

sigma (universality) = standard deviation (population) of the token-lookup F1 across pairs.

Writes results/baseline_f1.csv and results/baseline_f1.md.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import pandas as pd

from common import RESULTS_DIR, available_pairs, load_split, markdown_table
from lid import PAIRS

ALPHA = 2.0  # smoothing strength toward the prior
THRESHOLDS = [i / 20 for i in range(1, 20)]  # 0.05 ... 0.95


import re

_PUNCT_EDGES = re.compile(r"^[^\w]+|[^\w]+$")


def key_of(word: str, strip_punct: bool = False) -> str:
    """Lookup key: lowercase. Punctuation is kept by default because a trailing comma or
    period is a real cue for an upcoming switch; `strip_punct=True` removes it to measure
    how much of the baseline's score comes from punctuation alone."""
    w = word.lower()
    return _PUNCT_EDGES.sub("", w) or w if strip_punct else w


def prf(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def positions(convs, strip_punct: bool = False):
    """Yield (token_key, lang, y_sw, y_dur) for every labelled position."""
    for c in convs:
        for w, lang, sw, dur in zip(c["words"], c["langs"], c["y_sw"], c["y_dur"]):
            if sw >= 0:
                yield key_of(w, strip_punct), lang, sw, dur


class TokenLookup:
    """P(switch | current token) and P(bin | current token) from the training split."""

    def __init__(self, train, strip_punct: bool = False):
        self.n = Counter()
        self.sw = Counter()
        self.lang_n = Counter()
        self.lang_sw = Counter()
        self.bins = defaultdict(Counter)
        total = total_sw = 0
        for tok, lang, y, dur in positions(train, strip_punct):
            self.n[tok] += 1
            self.sw[tok] += y
            self.lang_n[lang] += 1
            self.lang_sw[lang] += y
            total += 1
            total_sw += y
            if y == 1:
                self.bins[tok][dur] += 1
        self.prior = total_sw / max(total, 1)
        all_bins = Counter()
        for c in self.bins.values():
            all_bins.update(c)
        self.majority_bin = all_bins.most_common(1)[0][0] if all_bins else 0

    def p_token(self, tok: str) -> float:
        return (self.sw[tok] + ALPHA * self.prior) / (self.n[tok] + ALPHA)

    def p_lang(self, lang: str) -> float:
        return (self.lang_sw[lang] + ALPHA * self.prior) / (self.lang_n[lang] + ALPHA)

    def bin_token(self, tok: str) -> int:
        c = self.bins.get(tok)
        return c.most_common(1)[0][0] if c else self.majority_bin


def tune_threshold(scores: list[float], y: list[int]) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLDS:
        f1 = prf(y, [1 if s >= t else 0 for s in scores])[2]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def evaluate_pair(pair: str) -> dict:
    train, dev, test = (load_split(pair, s) for s in ("train", "dev", "test"))
    model = TokenLookup(train)

    dev_pos = list(positions(dev))
    test_pos = list(positions(test))
    y_dev = [p[2] for p in dev_pos]
    y_test = [p[2] for p in test_pos]

    # token lookup
    t_tok = tune_threshold([model.p_token(p[0]) for p in dev_pos], y_dev)
    pred_tok = [1 if model.p_token(p[0]) >= t_tok else 0 for p in test_pos]
    P, R, F = prf(y_test, pred_tok)
    # token lookup without punctuation (how much of the score is the trailing comma or period?)
    model_np = TokenLookup(train, strip_punct=True)
    dev_np, test_np = list(positions(dev, True)), list(positions(test, True))
    t_np = tune_threshold([model_np.p_token(p[0]) for p in dev_np], [p[2] for p in dev_np])
    F_np = prf([p[2] for p in test_np], [1 if model_np.p_token(p[0]) >= t_np else 0 for p in test_np])[2]
    # language prior
    t_lang = tune_threshold([model.p_lang(p[1]) for p in dev_pos], y_dev)
    pred_lang = [1 if model.p_lang(p[1]) >= t_lang else 0 for p in test_pos]
    F_lang = prf(y_test, pred_lang)[2]
    # trivial references
    F_always = prf(y_test, [1] * len(y_test))[2]
    # duration at true switches
    sw_pos = [p for p in test_pos if p[2] == 1]
    acc_major = sum(1 for p in sw_pos if p[3] == model.majority_bin) / max(len(sw_pos), 1)
    acc_tok = sum(1 for p in sw_pos if p[3] == model.bin_token(p[0])) / max(len(sw_pos), 1)

    return {
        "pair": pair,
        "L1": PAIRS[pair][1],
        "test_positions": len(y_test),
        "switch_rate_%": round(100 * sum(y_test) / max(len(y_test), 1), 2),
        "F1_never": 0.0,
        "F1_always": round(F_always, 3),
        "F1_lang_prior": round(F_lang, 3),
        "threshold_token": t_tok,
        "P_token": round(P, 3),
        "R_token": round(R, 3),
        "F1_token": round(F, 3),
        "F1_token_nopunct": round(F_np, 3),
        "dur_acc_majority": round(acc_major, 3),
        "dur_acc_token": round(acc_tok, 3),
        "majority_bin": model.majority_bin,
    }


def main() -> None:
    rows = [evaluate_pair(p) for p in available_pairs()]
    df = pd.DataFrame(rows)
    f1 = df["F1_token"]
    mean, sigma = f1.mean(), f1.std(ddof=0)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "baseline_f1.csv", index=False)

    cols = ["pair", "switch_rate_%", "F1_never", "F1_always", "F1_lang_prior", "P_token", "R_token", "F1_token", "F1_token_nopunct",
            "threshold_token", "dur_acc_majority", "dur_acc_token"]
    table = markdown_table(cols, df[cols].values.tolist())
    np_f1 = df["F1_token_nopunct"]
    summary = (
        f"\nToken-lookup baseline over {len(df)} pairs: mean F1 = {mean:.3f}, sigma = {sigma:.3f} "
        f"(min {f1.min():.3f} {df.loc[f1.idxmin(), 'pair']}, max {f1.max():.3f} {df.loc[f1.idxmax(), 'pair']}).\n"
        f"Same lookup with punctuation stripped from the token: mean F1 = {np_f1.mean():.3f}, sigma = {np_f1.std(ddof=0):.3f}.\n"
        f"'always' reference: mean F1 = {df['F1_always'].mean():.3f}. Duration accuracy, majority bin: mean {df['dur_acc_majority'].mean():.3f}; "
        f"token bin: mean {df['dur_acc_token'].mean():.3f}."
    )
    (RESULTS_DIR / "baseline_f1.md").write_text("# Naive baselines, Anticipatory F1 on the test split\n\n" + table + "\n" + summary + "\n", encoding="utf-8")
    print(table)
    print(summary)
    print(f"\nwrote {RESULTS_DIR / 'baseline_f1.csv'} and baseline_f1.md")


if __name__ == "__main__":
    main()
