"""List dev and test conversations that duplicate an earlier split, so evaluation never sees a training text.

update1/split.py splits by conversation id, and a few conversations occur twice in SwitchLingua under
different ids. A dev conversation is listed when its word sequence also occurs in train; a test
conversation when it also occurs in train or dev. load_convs() drops the listed ids (data.py).

    python dedup.py            # writes duplicates.json next to this file
"""
from __future__ import annotations

import argparse
import json

import setup_path  # noqa: F401
from common import load_jsonl
from data import DEFAULT_DATA_DIR, DUPLICATES_FILE, available_pairs, split_path


def text_key(conv: dict) -> str:
    return "\u0001".join(conv["words"])


def find_duplicates(data_dir) -> dict:
    out = {"dev": [], "test": []}
    for pair in available_pairs(data_dir):
        seen = {text_key(c) for c in load_jsonl(split_path(data_dir, pair, "train"))}
        dev = load_jsonl(split_path(data_dir, pair, "dev"))
        out["dev"] += [c["conv_id"] for c in dev if text_key(c) in seen]
        seen |= {text_key(c) for c in dev}
        out["test"] += [c["conv_id"] for c in load_jsonl(split_path(data_dir, pair, "test")) if text_key(c) in seen]
    return {k: sorted(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()
    dup = find_duplicates(args.data_dir)
    DUPLICATES_FILE.write_text(json.dumps(dup, indent=1) + "\n", encoding="utf-8")
    print(f"dev: {len(dup['dev'])} duplicates of train; test: {len(dup['test'])} duplicates of train or dev -> {DUPLICATES_FILE}")


if __name__ == "__main__":
    main()
