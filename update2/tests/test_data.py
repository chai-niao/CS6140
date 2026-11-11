"""duplicates.json handling: dev / test conversations that repeat an earlier split are dropped."""
import json

import data


def _write_split(dir_, pair, split, convs):
    (dir_ / f"{pair}__{split}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in convs), encoding="utf-8")


def test_duplicates_are_found_and_dropped(tmp_path, monkeypatch):
    pair = "Spanish_eng"
    conv = lambda cid, words: {"conv_id": cid, "words": words}  # noqa: E731
    _write_split(tmp_path, pair, "train", [conv("Spanish_eng_1", ["hola", "you"]), conv("Spanish_eng_2", ["a", "b"])])
    _write_split(tmp_path, pair, "dev", [conv("Spanish_eng_3", ["a", "b"]), conv("Spanish_eng_4", ["c"])])
    _write_split(tmp_path, pair, "test", [conv("Spanish_eng_5", ["c"]), conv("Spanish_eng_6", ["hola", "you"]),
                                          conv("Spanish_eng_7", ["new"])])
    import dedup

    dup = dedup.find_duplicates(tmp_path)
    assert dup == {"dev": ["Spanish_eng_3"], "test": ["Spanish_eng_5", "Spanish_eng_6"]}

    dup_file = tmp_path / "duplicates.json"
    dup_file.write_text(json.dumps(dup), encoding="utf-8")
    monkeypatch.setattr(data, "DUPLICATES_FILE", dup_file)
    assert [c["conv_id"] for c in data.load_convs(tmp_path, "test", [pair])] == ["Spanish_eng_7"]
    assert len(data.load_convs(tmp_path, "test", [pair], drop_duplicates=False)) == 3
    assert len(data.load_convs(tmp_path, "train", [pair])) == 2


def test_duration_weight_rejects_unknown_modes():
    import pytest

    convs = [{"y_dur": [0, 1, 2, -1]}]
    assert data.duration_class_weight(convs, "none") is None
    assert len(data.duration_class_weight(convs, "balanced")) == 3
    for bad in ("2", "balance", "uniform"):
        with pytest.raises(ValueError):
            data.duration_class_weight(convs, bad)
