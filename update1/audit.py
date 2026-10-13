"""Draw a random sample of tagged words for a manual accuracy check of the language tagger.

Writes results/lid_audit_sample.csv with columns
  pair, method, word, context (5 words around it), predicted, correct (filled in manually)
Default: 20 words from every pair (300 rows). Tagging accuracy per method is the share of
rows marked correct.
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

from common import RESULTS_DIR, available_pairs, load_split
from lid import PAIRS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_pair", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for pair in available_pairs():
        convs = load_split(pair, "dev")
        for _ in range(args.per_pair):
            c = rng.choice(convs)
            i = rng.randrange(len(c["words"]))
            lo, hi = max(0, i - 2), min(len(c["words"]), i + 3)
            ctx = " ".join(("[" + w + "]") if j == i else w for j, w in enumerate(c["words"][lo:hi], start=lo))
            rows.append({"pair": pair, "method": PAIRS[pair][3], "word": c["words"][i], "context": ctx,
                         "predicted": c["langs"][i], "correct": ""})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "lid_audit_sample.csv"
    df.to_csv(out, index=False)
    print(df.groupby("method").size().to_string())
    print(f"\nwrote {out}  ({len(df)} rows). Mark the 'correct' column with y/n to compute accuracy.")


if __name__ == "__main__":
    main()
