import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import best_threshold, duration_accuracy, pair_rows, prf, summarize, tune_thresholds  # noqa: E402


def test_precision_recall_f1():
    y = np.array([1, 1, 0, 0, 1])
    pred = np.array([1, 0, 0, 1, 1])
    p, r, f1 = prf(y, pred)
    assert abs(p - 2 / 3) < 1e-9 and abs(r - 2 / 3) < 1e-9 and abs(f1 - 2 / 3) < 1e-9
    assert prf(y, np.zeros(5, int)) == (0.0, 0.0, 0.0)


def test_best_threshold_separates_the_classes():
    p = np.array([0.9, 0.2, 0.6, 0.1])
    y = np.array([1, 0, 1, 0])
    t, f1 = best_threshold(p, y)
    assert f1 == 1.0 and 0.2 < t <= 0.6


def test_duration_accuracy_counts_only_true_switches():
    y_sw = np.array([1, 0, 1, 1])
    y_dur = np.array([0, -1, 2, 1])
    pred = np.array([0, 2, 1, 1])  # the prediction at the non-switch position must not count
    assert abs(duration_accuracy(pred, y_dur, y_sw) - 2 / 3) < 1e-9


def records(pair_a_scale=1.0):
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, 2 * n)
    p = np.clip(0.3 * rng.standard_normal(2 * n) + 0.7 * y, 0, 1).astype(np.float32)
    p[:n] = np.clip(p[:n] * pair_a_scale, 0, 1)
    return {
        "pair": np.array(["A_eng"] * n + ["B_eng"] * n, dtype=object),
        "cs_type": np.array((["Intrasentential", "Intersentential"] * n), dtype=object),
        "p_switch": p,
        "y_sw": y.astype(np.int8),
        "dur_pred": rng.integers(0, 3, 2 * n).astype(np.int8),
        "y_dur": np.where(y == 1, rng.integers(0, 3, 2 * n), -1).astype(np.int8),
    }


def test_pair_rows_and_sigma():
    rec = records()
    thresholds = tune_thresholds(rec)
    rows = pair_rows(rec, thresholds, lambda t: t[:5].lower())
    assert [r["pair"] for r in rows] == ["A_eng", "B_eng"]
    for r in rows:
        assert 0 <= r["F1"] <= 1 and 0 <= r["dur_acc"] <= 1
        assert "F1_intra" in r and "F1_inter" in r
        assert abs(r["F1_macro_by_type"] - np.mean([r["F1_intra"], r["F1_inter"]])) < 1e-3
    s = summarize(rows)
    f1 = np.array([r["F1"] for r in rows])
    assert abs(s["mean_F1"] - f1.mean()) < 1e-3
    assert abs(s["sigma_F1"] - f1.std(ddof=0)) < 1e-3
    assert s["min_pair"] in ("A_eng", "B_eng") and s["pairs"] == 2
