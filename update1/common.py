"""Small helpers shared by the Update 1 scripts (no torch dependency)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROCESSED_DIR = HERE / "data_processed"                  # output of split.py
RESULTS_DIR = HERE / "results"
SPLITS = ("train", "dev", "test")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(pair: str, split: str, processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    return load_jsonl(processed_dir / pair / f"{split}.jsonl")


def available_pairs(processed_dir: Path = PROCESSED_DIR) -> list[str]:
    return sorted(p.name for p in processed_dir.iterdir() if (p / "train.jsonl").exists())


def markdown_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)
