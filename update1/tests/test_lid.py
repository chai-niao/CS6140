import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lid import PAIRS, tag_by_frequency, tag_by_script, tokenize_and_tag  # noqa: E402


def langs(text, pair):
    return [l for _, l in tokenize_and_tag(text, pair)]


def test_all_15_pairs_are_configured():
    assert len(PAIRS) == 15
    assert {m for _, _, _, m in PAIRS.values()} == {"script", "cjk", "freq", "hinglish"}


def test_romanized_hindi():
    got = tokenize_and_tag("Woh kal Prime Minister se milne wali hain, to discuss the new tariffs.", "Hindi_eng")
    tags = dict(got)
    assert tags["Woh"] == "hi" and tags["kal"] == "hi" and tags["milne"] == "hi" and tags["wali"] == "hi" and tags["hain,"] == "hi"
    assert tags["Prime"] == "en" and tags["Minister"] == "en" and tags["discuss"] == "en" and tags["tariffs."] == "en"
    assert tags["se"] == "hi"  # ambiguous word between an English and a Hindi neighbour: tie goes to Hindi


def test_devanagari_hindi_still_works():
    got = tokenize_and_tag("मुझे लगता है कि this project is interesting", "Hindi_eng")
    assert [l for _, l in got] == ["hi", "hi", "hi", "hi", "en", "en", "en", "en"]


def test_script_pairs():
    assert tag_by_script("الموضوع", "ar") == "ar"
    assert tag_by_script("honestly", "ar") == "en"
    assert tag_by_script("नमस्ते", "hi") == "hi"
    assert tag_by_script("안녕하세요", "ko") == "ko"
    assert tag_by_script("привет,", "ru") == "ru"
    assert tag_by_script("2024", "ru") is None  # digits only: dropped


def test_arabic_sentence():
    assert langs("أنا تابعت خبر الفيزا، honestly الموضوع معقد", "Arabic_eng") == ["ar", "ar", "ar", "ar", "en", "ar", "ar"]


def test_spanish_english_frequency():
    got = langs("Cuando termine mi maestría, I will invest in sustainable projects.", "Spanish_eng")
    assert got == ["es"] * 4 + ["en"] * 6


def test_sticky_rule_on_ambiguous_word():
    # "a" exists in both languages with similar frequency, so it follows the previous word
    assert tag_by_frequency("a", "es", "en", prev="es") == "es"
    assert tag_by_frequency("a", "es", "en", prev="en") == "en"


def test_diacritics_decide():
    assert tag_by_frequency("niño", "es", "en", prev="en") == "es"
    assert tag_by_frequency("straße", "de", "fr", prev="fr") == "de"
    assert tag_by_frequency("ça", "de", "fr", prev="de") == "fr"


def test_cjk_segmentation_and_tagging():
    got = tokenize_and_tag("我今天下午有个meeting，你先帮我review一下。", "Chinese_eng")
    tags = [l for _, l in got]
    assert "en" in tags and "zh" in tags
    assert [w for w, l in got if l == "en"] == ["meeting", "review"]
    # punctuation pieces from the segmenter are dropped
    assert all(any(c.isalpha() for c in w) for w, _ in got)
