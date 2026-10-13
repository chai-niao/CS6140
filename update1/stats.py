"""Step 2: descriptive statistics of the labelled data for every pair.

Writes results/pair_stats.csv and results/pair_stats.md with, per pair:
  conversations, words, words per conversation, share of L2 words,
  switch rate (share of positions with y_sw = 1), burst-length bin shares,
  and the switch rate broken down by the dataset's cs_type field.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from common import RESULTS_DIR, SPLITS, available_pairs, load_split, markdown_table
from lid import PAIRS


def pair_stats(pair: str) -> dict:
    l1 = PAIRS[pair][0]
    n_conv = n_words = n_pos = n_sw = n_l2 = 0
    bins = Counter()
    by_type = {}
    for split in SPLITS:
        for conv in load_split(pair, split):
            n_conv += 1
            n_words += len(conv["words"])
            n_l2 += sum(1 for l in conv["langs"] if l != l1)
            sw = [v for v in conv["y_sw"] if v >= 0]
            n_pos += len(sw)
            n_sw += sum(sw)
            bins.update(v for v in conv["y_dur"] if v >= 0)
            bt = by_type.setdefault(conv["cs_type"], [0, 0])
            bt[0] += len(sw)
            bt[1] += sum(sw)
    total_sw = sum(bins.values()) or 1
    row = {
        "pair": pair,
        "L1": PAIRS[pair][1],
        "L2": "French" if pair == "German_French" else "English",
        "conversations": n_conv,
        "words": n_words,
        "words_per_conv": round(n_words / max(n_conv, 1), 1),
        "L2_share_%": round(100 * n_l2 / max(n_words, 1), 1),
        "switch_rate_%": round(100 * n_sw / max(n_pos, 1), 2),
        "switches_per_conv": round(n_sw / max(n_conv, 1), 2),
        "small_%": round(100 * bins[0] / total_sw, 1),
        "medium_%": round(100 * bins[1] / total_sw, 1),
        "large_%": round(100 * bins[2] / total_sw, 1),
    }
    for t, (pos, sw) in sorted(by_type.items()):
        key = {"Extra-sentential / Tag switching": "tag", "Intersentential": "inter", "Intrasentential": "intra"}.get(t, t)
        row[f"switch_rate_{key}_%"] = round(100 * sw / max(pos, 1), 2)
    return row


def main() -> None:
    rows = [pair_stats(p) for p in available_pairs()]
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "pair_stats.csv", index=False)

    cols = ["pair", "conversations", "words_per_conv", "L2_share_%", "switch_rate_%", "small_%", "medium_%", "large_%",
            "switch_rate_tag_%", "switch_rate_inter_%", "switch_rate_intra_%"]
    table = markdown_table(cols, df[cols].values.tolist())
    summary = (
        f"\nTotals: {df['conversations'].sum():,} conversations, {df['words'].sum():,} words. "
        f"Switch rate mean {df['switch_rate_%'].mean():.2f}% (min {df['switch_rate_%'].min():.2f}%, max {df['switch_rate_%'].max():.2f}%)."
    )
    (RESULTS_DIR / "pair_stats.md").write_text("# Per-pair statistics (train + dev + test)\n\n" + table + "\n" + summary + "\n", encoding="utf-8")
    print(table)
    print(summary)
    print(f"\nwrote {RESULTS_DIR / 'pair_stats.csv'} and pair_stats.md")


if __name__ == "__main__":
    main()
