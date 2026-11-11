"""Small helpers shared by the Update 1 scripts (no torch dependency)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


# ------------------------------------------------------------------ switch F1 and threshold tuning
# One implementation for the naive baselines (baselines.py) and the trained models (update2/metrics.py),
# so that both are scored with exactly the same F1 and the same threshold grid.
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95


def prf(y_true, y_pred) -> tuple[float, float, float]:
    """Precision, recall and F1 of the positive class (1)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def f1_at(p, y, threshold: float) -> float:
    return prf(y, (np.asarray(p) >= threshold).astype(np.int8))[2]


def best_threshold(p, y) -> tuple[float, float]:
    """(threshold, F1) maximising F1 over THRESHOLDS; ties keep the lowest threshold, 0.5 if nothing scores."""
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLDS:
        f1 = f1_at(p, y, t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1
