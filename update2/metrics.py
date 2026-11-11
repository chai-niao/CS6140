"""Evaluation metrics (project spec, section III).

Anticipatory F1   F1 of the switch prediction y_sw(t) over all labelled positions of a pair.
                  The decision threshold on P(switch) is tuned per pair on the dev split and
                  frozen for the test split, exactly like the naive baseline of Update 1.
Duration accuracy accuracy of the burst-length bin at the positions where a switch really happens.
Macro by type     F1 computed separately for the dataset's three switch types (intra-sentential,
                  inter-sentential, tag) and averaged.
Universality      sigma = population standard deviation of the per-pair F1 (smaller = more universal).

`records` is a dict of equally long numpy arrays with one entry per labelled position:
  pair, cs_type, p_switch, y_sw, dur_pred, y_dur
"""
from __future__ import annotations

import numpy as np

import setup_path  # noqa: F401

# F1 and the threshold grid are shared with the naive baselines of Update 1
from common import THRESHOLDS, best_threshold, f1_at, prf  # noqa: E402,F401


def tune_thresholds(records: dict, pairs: list[str] | None = None) -> dict[str, float]:
    pairs = pairs or sorted(set(records["pair"].tolist()))
    out = {}
    for pair in pairs:
        m = records["pair"] == pair
        out[pair] = best_threshold(records["p_switch"][m], records["y_sw"][m])[0] if m.any() else 0.5
    return out


def duration_accuracy(dur_pred: np.ndarray, y_dur: np.ndarray, y_sw: np.ndarray) -> float:
    at_switch = y_sw == 1
    if not at_switch.any():
        return float("nan")
    return float((dur_pred[at_switch] == y_dur[at_switch]).mean())


def pair_rows(records: dict, thresholds: dict[str, float], type_short) -> list[dict]:
    """One row of metrics per pair. `type_short` maps a cs_type string to a short column suffix."""
    rows = []
    for pair in sorted(set(records["pair"].tolist())):
        m = records["pair"] == pair
        p, y = records["p_switch"][m], records["y_sw"][m]
        thr = thresholds.get(pair, 0.5)
        P, R, F = prf(y, (p >= thr).astype(np.int8))
        row = {
            "pair": pair,
            "positions": int(m.sum()),
            "switch_rate_%": round(100 * float(y.mean()), 2),
            "threshold": thr,
            "P": round(P, 4),
            "R": round(R, 4),
            "F1": round(F, 4),
            "F1_at_0.5": round(f1_at(p, y, 0.5), 4),
            "dur_acc": round(duration_accuracy(records["dur_pred"][m], records["y_dur"][m], y), 4),
        }
        by_type = {}
        for cs_type in sorted(set(records["cs_type"][m].tolist())):
            mt = records["cs_type"][m] == cs_type
            by_type[type_short(cs_type)] = f1_at(p[mt], y[mt], thr)
        row["F1_macro_by_type"] = round(float(np.mean(list(by_type.values()))), 4) if by_type else float("nan")
        for name, value in sorted(by_type.items()):
            row[f"F1_{name}"] = round(value, 4)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Mean and spread over pairs of the per-pair rows."""
    f1 = np.array([r["F1"] for r in rows], dtype=float)
    i_min, i_max = int(f1.argmin()), int(f1.argmax())
    out = {
        "pairs": len(rows),
        "mean_F1": round(float(f1.mean()), 4),
        "sigma_F1": round(float(f1.std(ddof=0)), 4),
        "min_F1": round(float(f1[i_min]), 4),
        "min_pair": rows[i_min]["pair"],
        "max_F1": round(float(f1[i_max]), 4),
        "max_pair": rows[i_max]["pair"],
        "mean_F1_at_0.5": round(float(np.mean([r["F1_at_0.5"] for r in rows])), 4),
        "mean_dur_acc": round(float(np.nanmean([r["dur_acc"] for r in rows])), 4),
        "mean_F1_macro_by_type": round(float(np.nanmean([r["F1_macro_by_type"] for r in rows])), 4),
    }
    for key in sorted({k for r in rows for k in r if k.startswith("F1_") and k not in ("F1_at_0.5", "F1_macro_by_type")}):
        out[f"mean_{key}"] = round(float(np.nanmean([r.get(key, np.nan) for r in rows])), 4)
    return out
