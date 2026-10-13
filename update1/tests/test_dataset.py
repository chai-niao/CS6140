import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("transformers")
pytest.importorskip("torch")

from dataset import IGNORE, PrefixWindowDataset, bos_id, encode_conversation, encode_words, get_tokenizer, iter_prefixes  # noqa: E402
from labels import make_labels  # noqa: E402
from lid import tokenize_and_tag_spaced  # noqa: E402

SENTENCES = [
    ("Chinese_eng", "我今天下午有个meeting，你先帮我review一下那个文件。"),
    ("Japanese_eng", "今日はmeetingがあるので、ちょっと忙しいです。"),
    ("Spanish_eng", "Ayer leí una noticia sobre el conflicto en Yemen. The news was sad."),
    ("Korean_eng", "오늘 referee를 만났어요, honestly 좋았어요."),
]


def normalized(tok, text):
    """XLM-R normalizes full-width punctuation (NFKC): "，" -> ","."""
    norm = tok.backend_tokenizer.normalizer
    return norm.normalize_str(text) if norm else text


def make_conv(pair, text):
    trip = tokenize_and_tag_spaced(text, pair)
    langs = [l for _, l, _ in trip]
    y_sw, y_dur = make_labels(langs)
    return {"pair": pair, "conv_id": "t", "words": [w for w, _, _ in trip], "langs": langs,
            "space_before": [s for _, _, s in trip], "y_sw": y_sw, "y_dur": y_dur}


@pytest.fixture(scope="module", params=["xlmr", "qwen"])
def tok(request):
    try:
        return get_tokenizer(request.param)
    except OSError as e:  # offline and not cached
        pytest.skip(f"tokenizer not available: {e}")


@pytest.mark.parametrize("pair,text", SENTENCES)
def test_word_by_word_encoding_keeps_the_original_text(tok, pair, text):
    # no spaces are invented between Chinese or Japanese words, and nothing is dropped
    conv = make_conv(pair, text)
    ids, _ = encode_words(conv["words"], conv["space_before"], tok)
    assert tok.decode(ids).strip() == normalized(tok, text)


def test_no_subword_spans_two_words(tok):
    # the natural tokenization merges 帮+我 into one token; per-word encoding must not
    conv = make_conv("Chinese_eng", SENTENCES[0][1])
    ids, wids = encode_words(conv["words"], conv["space_before"], tok)
    for w, word in enumerate(conv["words"]):
        piece = [i for i, wid in zip(ids, wids) if wid == w]
        assert tok.decode(piece).strip() == normalized(tok, word)


@pytest.mark.parametrize("pair,text", SENTENCES)
def test_labels_sit_on_last_subword(tok, pair, text):
    conv = make_conv(pair, text)
    e = encode_conversation(conv, tok)
    for i, w in enumerate(e["word_ids"]):
        is_last = i + 1 == len(e["word_ids"]) or e["word_ids"][i + 1] != w
        if is_last:
            assert e["y_sw"][i] == conv["y_sw"][w] and e["y_dur"][i] == conv["y_dur"][w]
        else:
            assert e["y_sw"][i] == IGNORE and e["y_dur"][i] == IGNORE


@pytest.mark.parametrize("pair,text", SENTENCES)
def test_streaming_prefix_equals_training_window(tok, pair, text):
    # same BOS and same ids: the word-by-word view is a prefix of the training input
    conv = make_conv(pair, text)
    window = PrefixWindowDataset([conv], tok, max_len=128, stride=64)[0]
    for w, prefix, y_sw, _ in iter_prefixes(conv, tok):
        assert window["input_ids"][: len(prefix)] == prefix
        assert window["y_sw"][len(prefix) - 1] == y_sw
    if bos_id(tok) is not None:
        assert window["input_ids"][0] == bos_id(tok)
