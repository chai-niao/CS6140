import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labels import IGNORE, burst_bin, make_labels  # noqa: E402


def test_bins():
    assert [burst_bin(n) for n in (1, 2, 3, 6, 7, 20)] == [0, 0, 1, 1, 2, 2]


def test_single_switch_medium_run():
    # "Cuando termine mi maestría, | I will invest in sustainable projects."
    langs = ["es"] * 4 + ["en"] * 6
    y_sw, y_dur = make_labels(langs)
    assert y_sw == [0, 0, 0, 1, 0, 0, 0, 0, 0, IGNORE]
    assert y_dur == [IGNORE, IGNORE, IGNORE, 1, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE]


def test_two_switches_small_then_large():
    # es es | en en | es es es es es es es  -> at t=1 switch with run 2 (Small), at t=3 switch with run 7 (Large)
    langs = ["es", "es", "en", "en"] + ["es"] * 7
    y_sw, y_dur = make_labels(langs)
    assert y_sw[1] == 1 and y_dur[1] == 0
    assert y_sw[3] == 1 and y_dur[3] == 2
    assert sum(1 for v in y_sw if v == 1) == 2
    assert y_sw[-1] == IGNORE and y_dur[-1] == IGNORE


def test_no_switch_stream():
    y_sw, y_dur = make_labels(["ar"] * 5)
    assert y_sw == [0, 0, 0, 0, IGNORE]
    assert all(v == IGNORE for v in y_dur)


def test_empty_and_singleton():
    assert make_labels([]) == ([], [])
    assert make_labels(["en"]) == ([IGNORE], [IGNORE])
