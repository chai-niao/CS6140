"""End-to-end smoke test: a tiny random backbone, two pairs, a handful of steps on the CPU."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch")
pytest.importorskip("transformers")

import setup_path  # noqa: E402, F401
from data import DEFAULT_DATA_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(not (DEFAULT_DATA_DIR / "Spanish_eng" / "train.jsonl").exists(),
                                reason="run update1/split.py first")


@pytest.mark.parametrize("model", ["xlmr", "qwen"])
def test_train_evaluate_roundtrip(tmp_path, model):
    import train
    import evaluate

    try:
        summary = train.main([
            "--tiny", "--model", model, "--pairs", "Spanish_eng", "Hindi_eng",
            "--max_train_convs", "20", "--max_dev_convs", "10", "--max_test_convs", "10",
            "--epochs", "1", "--max_steps", "4", "--batch_size", "4", "--eval_every", "2", "--log_every", "1",
            "--device", "cpu", "--out_dir", str(tmp_path), "--run_name", "smoke",
        ])
    except OSError as e:  # tokenizer not cached and offline
        pytest.skip(f"tokenizer not available: {e}")
    for name in ("config.json", "train.log", "train_log.csv", "dev_log.csv", "dev_metrics.csv", "test_metrics.csv",
                 "summary.json", "results.md", "model.pt"):
        assert (tmp_path / name).exists(), name
    assert summary["steps"] == 4
    assert summary["test"]["pairs"] == 2
    assert 0 <= summary["test"]["mean_F1"] <= 1
    assert all(v == v for v in summary["test_losses"].values())  # no NaN
    log_lines = (tmp_path / "train_log.csv").read_text().strip().splitlines()
    assert len(log_lines) == 1 + 4  # header + one row per step (log_every 1)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["tiny"] is True

    res = evaluate.main(["--run_dir", str(tmp_path), "--split", "test", "--max_convs", "5", "--device", "cpu",
                         "--streaming_check", "2"])
    assert res["streaming_max_abs_diff"] < 1e-3
