import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lid import PAIRS, tag_by_frequency, tag_by_script, tokenize_and_tag, tokenize_and_tag_spaced  # noqa: E402


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
    assert [w for w, l in got if l == "en"] == ["meeting，", "review"]
    # punctuation pieces from the segmenter are attached to the previous word, not dropped
    assert all(any(c.isalpha() for c in w) for w, _ in got)
    assert got[-1] == ("一下。", "zh")


def test_cjk_text_is_rebuilt_exactly():
    # space_before + words reproduce the sentence (no characters dropped, no spaces invented)
    for pair, text in [
        ("Chinese_eng", "我今天下午有个meeting，你先帮我review一下那个文件。"),
        ("Can_Eng", "「佢話」OK la，唔使擔心喇。"),
        ("Japanese_eng", "今日はmeetingがあるので、ちょっと忙しいです。"),
        ("Spanish_eng", "¿Qué tal? I am fine."),
    ]:
        got = tokenize_and_tag_spaced(text, pair)
        rebuilt = "".join((" " if sp and i else "") + w for i, (w, _, sp) in enumerate(got))
        assert rebuilt == text


def test_hinglish_english_homographs():
    # mile, log, tab, mere are also Hindi words; in an English sentence they must stay English
    got = langs("I walked a mile to log in at the tab, mere coincidence.", "Hindi_eng")
    assert got == ["en"] * 12
    got = dict(tokenize_and_tag("Mere dost ne kaha ki woh kal aayega.", "Hindi_eng"))
    assert got["Mere"] == "hi" and got["dost"] == "hi"


def test_cantonese_segmenter_mapping_does_not_grow():
    # pycantonese gives every unseen ASCII token a new character chr(0xE000 + n); once n
    # passed the Private Use Area, "，" decoded as "Kevon". Only CJK text may reach it.
    from pycantonese.word_segmentation.segmenter import _get_default_segmenter

    tokenize_and_tag("暖身", "Can_Eng")
    before = len(_get_default_segmenter()._alphanum_to_pua)
    tokenize_and_tag("我睇咗Zorblaxquon同Quintessa-7777嘅news，真係amazingzz！", "Can_Eng")
    assert len(_get_default_segmenter()._alphanum_to_pua) == before


def test_latin_runs_inside_cjk_chunks_stay_whole():
    got = tokenize_and_tag("我用iPhone15同V3模型，好正。", "Can_Eng")
    assert [w for w, l in got if l == "en"] == ["iPhone15", "V3"]
