"""Loading the labelled conversations of Update 1 and the class weights for the loss.

Expected layout (output of update1/split.py):   <data_dir>/<pair>/{train,dev,test}.jsonl
A flat layout <data_dir>/<pair>__<split>.jsonl is accepted as well.

Every conversation is a dict with words, langs, space_before, y_sw, y_dur, cs_type, ...
(see update1/split.py). Windows and subword alignment come from update1/dataset.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import setup_path  # noqa: F401
from common import load_jsonl
from lid import PAIRS

DEFAULT_DATA_DIR = setup_path.UPDATE1_DIR / "data_processed"
# dev / test conversations whose text also occurs in an earlier split (written by dedup.py)
DUPLICATES_FILE = setup_path.UPDATE2_DIR / "duplicates.json"
SPLITS = ("train", "dev", "test")

# short names of the dataset's cs_type values, used as column suffixes
SWITCH_TYPES = {
    "Intrasentential": "intra",
    "Intersentential": "inter",
    "Extra-sentential / Tag switching": "tag",
}


def split_path(data_dir: Path | str, pair: str, split: str) -> Path:
    nested = Path(data_dir) / pair / f"{split}.jsonl"
    if nested.exists():
        return nested
    flat = Path(data_dir) / f"{pair}__{split}.jsonl"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"no {split} split for {pair} under {data_dir}")


def available_pairs(data_dir: Path | str) -> list[str]:
    out = []
    for pair in PAIRS:
        try:
            split_path(data_dir, pair, "train")
        except FileNotFoundError:
            continue
        out.append(pair)
    return out


def duplicate_ids(split: str) -> set[str]:
    """Ids of `split` conversations that duplicate a train (or, for test, a dev) conversation."""
    if split == "train" or not DUPLICATES_FILE.exists():
        return set()
    return set(json.loads(DUPLICATES_FILE.read_text(encoding="utf-8")).get(split, []))


def load_convs(data_dir: Path | str, split: str, pairs: list[str], max_per_pair: int = 0, seed: int = 0,
               drop_duplicates: bool = True) -> list[dict]:
    """Conversations of `pairs` in one list. `max_per_pair` > 0 takes a seeded random subset of every pair.
    Dev and test conversations listed in duplicates.json are dropped unless drop_duplicates is False."""
    drop = duplicate_ids(split) if drop_duplicates else set()
    out: list[dict] = []
    for pair in pairs:
        convs = [c for c in load_jsonl(split_path(data_dir, pair, split)) if c["conv_id"] not in drop]
        if max_per_pair and len(convs) > max_per_pair:
            convs = random.Random(seed).sample(convs, max_per_pair)
        out.extend(convs)
    return out


def pair_of(conv_id: str) -> str:
    """conv_id is '<pair>_<row>' (split.py); pair names themselves contain underscores."""
    return conv_id.rsplit("_", 1)[0]


def switch_type_short(cs_type: str) -> str:
    return SWITCH_TYPES.get(cs_type, cs_type.lower().replace(" ", "_"))


def switch_statistics(convs: list[dict]) -> tuple[int, int]:
    """(labelled positions, positions with a switch)."""
    n = pos = 0
    for c in convs:
        for v in c["y_sw"]:
            if v >= 0:
                n += 1
                pos += v
    return n, pos


def duration_counts(convs: list[dict]) -> list[int]:
    counts = [0, 0, 0]
    for c in convs:
        for v in c["y_dur"]:
            if v >= 0:
                counts[v] += 1
    return counts


def switch_class_weight(convs: list[dict], mode: str) -> list[float] | None:
    """Weight of the two switch classes [no switch, switch].

    'none'      no weighting
    'balanced'  switch class weighted by (1 - p) / p, p = switch rate of the training data,
                so that both classes contribute equally to the loss
    a number    weight of the switch class
    """
    if mode == "none":
        return None
    if mode == "balanced":
        n, pos = switch_statistics(convs)
        return [1.0, (n - pos) / max(pos, 1)]
    return [1.0, float(mode)]


def duration_class_weight(convs: list[dict], mode: str) -> list[float] | None:
    """'none', or 'balanced' = inverse bin frequency normalised to mean 1."""
    if mode == "none":
        return None
    counts = duration_counts(convs)
    total = sum(counts)
    return [total / (len(counts) * max(k, 1)) for k in counts]
