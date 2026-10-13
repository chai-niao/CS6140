"""Step 1 of the pipeline: raw SwitchLingua CSVs -> tagged, labelled, split JSONL files.

For every language pair:
  1. read the CSV and parse `data_generation_result` (a stringified list of sentences)
  2. cap the pair at --max_convs conversations (random, seeded) so that all pairs are
     comparable; this is what subsamples Cantonese from 119k to 8k
  3. tag every word with its language (lid.py) and generate y_sw / y_dur (labels.py)
  4. split 80 / 10 / 10 at the conversation level (no sentence leaks across splits)
  5. write data_processed/<pair>/{train,dev,test}.jsonl

Each JSONL row is one conversation:
  {pair, conv_id, cs_type, cs_ratio, conversation_type, score,
   words: [...], langs: [...], sent_ids: [...], y_sw: [...], y_dur: [...]}

Usage:
  python split.py                       # all 15 pairs
  python split.py --pairs Spanish_eng Hindi_eng
"""
from __future__ import annotations

import argparse
import ast
import random
import time

import pandas as pd
from tqdm import tqdm

from common import DATA_DIR, PROCESSED_DIR, markdown_table, write_jsonl
from labels import make_labels
from lid import PAIRS, tokenize_and_tag

COLUMNS = ["cs_type", "cs_ratio", "conversation_type", "score", "data_generation_result"]


def parse_sentences(cell) -> list[str]:
    """`data_generation_result` is a Python-literal list; items are strings or {'speaker','text'} dicts."""
    try:
        value = ast.literal_eval(cell)
    except Exception:
        return [str(cell)]
    if not isinstance(value, list):
        value = [value]
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = item.get("text") or " ".join(str(v) for v in item.values())
            out.append(str(text))
        elif isinstance(item, (list, tuple)):
            out.extend(str(x) for x in item)
        else:
            out.append(str(item))
    return [s for s in out if s.strip()]


def build_conversation(pair: str, conv_id: str, row) -> dict | None:
    words, langs, sent_ids = [], [], []
    for s_id, sent in enumerate(parse_sentences(row["data_generation_result"])):
        for w, lang in tokenize_and_tag(sent, pair):
            words.append(w)
            langs.append(lang)
            sent_ids.append(s_id)
    if len(words) < 2:
        return None
    y_sw, y_dur = make_labels(langs)
    return {
        "pair": pair,
        "conv_id": conv_id,
        "cs_type": row["cs_type"],
        "cs_ratio": row["cs_ratio"],
        "conversation_type": row["conversation_type"],
        "score": float(row["score"]),
        "words": words,
        "langs": langs,
        "sent_ids": sent_ids,
        "y_sw": y_sw,
        "y_dur": y_dur,
    }


def process_pair(pair: str, max_convs: int, seed: int) -> dict:
    df = pd.read_csv(DATA_DIR / f"{pair}.csv", usecols=COLUMNS, low_memory=False)
    order = list(range(len(df)))
    random.Random(seed).shuffle(order)
    order = order[:max_convs]

    convs = []
    for i in tqdm(order, desc=f"{pair:16s}", unit="conv", leave=False):
        conv = build_conversation(pair, f"{pair}_{i}", df.iloc[i])
        if conv is not None:
            convs.append(conv)

    n = len(convs)
    n_train, n_dev = int(0.8 * n), int(0.1 * n)
    splits = {"train": convs[:n_train], "dev": convs[n_train:n_train + n_dev], "test": convs[n_train + n_dev:]}
    for name, rows in splits.items():
        write_jsonl(PROCESSED_DIR / pair / f"{name}.jsonl", rows)
    return {"pair": pair, "raw_rows": len(df), "kept": n, **{k: len(v) for k, v in splits.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS), help="subset of pair file stems")
    ap.add_argument("--max_convs", type=int, default=8000, help="cap per pair (balances Cantonese)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    summary = []
    for pair in args.pairs:
        summary.append(process_pair(pair, args.max_convs, args.seed))
        s = summary[-1]
        print(f"{pair:16s} raw {s['raw_rows']:>7,}  kept {s['kept']:>5,}  train/dev/test {s['train']}/{s['dev']}/{s['test']}")
    print()
    print(markdown_table(["pair", "raw rows", "kept", "train", "dev", "test"],
                         [[s["pair"], f"{s['raw_rows']:,}", s["kept"], s["train"], s["dev"], s["test"]] for s in summary]))
    print(f"\nwrote {PROCESSED_DIR}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
