"""Word-level language identification for the 15 SwitchLingua_text language pairs.

SwitchLingua gives mixed-language sentences but no per-word language tags, so we
derive them here. Three strategies, chosen per pair:

  script  Arabic, Hindi, Korean, Russian: the Unicode block of the letters decides.
  cjk     Mandarin, Cantonese, Japanese: no spaces between words, so a segmenter
          (jieba / pycantonese / fugashi) splits the text first, then the script decides.
  freq    Spanish, French, German, Italian, Malay, Filipino, Vietnamese, German+French:
          both languages use the Latin alphabet, so we compare word frequencies from the
          `wordfreq` package, with diacritics as a tie breaker and the previous word's
          language as the fallback ("sticky" rule).

Public API:
    tokenize_and_tag(text, pair) -> list of (word, lang_code)
    PAIRS                        -> metadata for each CSV file stem
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# file stem -> (L1 code, L1 name, L2 code, method). L1 is the matrix language of the pair.
PAIRS: dict[str, tuple[str, str, str, str]] = {
    "Arabic_eng": ("ar", "Arabic", "en", "script"),
    "Hindi_eng": ("hi", "Hindi", "en", "hinglish"),  # 59% of conversations are romanized Hindi
    "Korean_eng": ("ko", "Korean", "en", "script"),
    "Russian_eng": ("ru", "Russian", "en", "script"),
    "Chinese_eng": ("zh", "Mandarin", "en", "cjk"),
    "Can_Eng": ("yue", "Cantonese", "en", "cjk"),
    "Japanese_eng": ("ja", "Japanese", "en", "cjk"),
    "Spanish_eng": ("es", "Spanish", "en", "freq"),
    "French_eng": ("fr", "French", "en", "freq"),
    "German_Eng": ("de", "German", "en", "freq"),
    "Italian_eng": ("it", "Italian", "en", "freq"),
    "Malay_eng": ("ms", "Malay", "en", "freq"),
    "Philippines_eng": ("fil", "Filipino", "en", "freq"),
    "Vietnamese_eng": ("vi", "Vietnamese", "en", "freq"),
    "German_French": ("de", "German", "fr", "freq"),
}

CJK_RANGES = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0xF900, 0xFAFF)]
SCRIPT_RANGES = {
    "ar": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "hi": [(0x0900, 0x097F)],
    "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "ru": [(0x0400, 0x04FF)],
    "zh": CJK_RANGES,
    "yue": CJK_RANGES,
    "ja": CJK_RANGES + [(0x3040, 0x309F), (0x30A0, 0x30FF), (0xFF66, 0xFF9F)],
}
LATIN_RANGES = [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)]

# letters that (almost) never occur in the other language of the pair
DIACRITIC_HINTS = {
    "es": set("ñáéíóúü¿¡"),
    "fr": set("àâçèêëîïôûùüÿœæ"),
    "de": set("äöüß"),
    "it": set("àèìòù"),
    "vi": set("ăâđêôơư"),
    "ms": set(),
    "fil": set(),
    "en": set(),
}
FREQ_MARGIN = 0.5  # zipf difference needed before frequency alone decides

# Romanized Hindi (Hinglish). SwitchLingua writes most Hindi in Latin letters, so the
# Devanagari script test alone would tag it as English. Words in SURE are Hindi whenever
# they appear; words in AMBIGUOUS are also ordinary English words, so they take the
# language of their neighbours.
HINGLISH_SURE = set("""
hai hain hoon hun tha thi thay kya kyun kyon nahi nahin nhi aur yeh ye woh wo mera meri mere tera teri tere
tumhara tumhari aapka aapki aap tum hum humne humko hume tumhe tumko mujhe mujhko mujh bhi toh abhi phir fir
bahut bohot bahot kuch sab sabhi koi karna karke karo karta karti karte kiya kiye raha rahi rahe rahega rahegi gaya
gayi gaye jaana jana jaa jao jaata jaati liye wala wali wale matlab achha accha acha theek thik haan haa lekin
magar agar kyunki kyuki jab tab wahan yahan kal aaj kaise kab kahan kitna kitni kitne kaun kaunsa sirf zyada
kam thoda thodi thode pata lagta lagti lagte lag dekho dekha dekhte suno suna bolo bola bolte kaha kehna kehte
chalo chal chalte milne milna mila mili mile ghar kaam paisa paise dost doston log logon baat baatein samajh
samjha samjhe yaar bhai arre arey bilkul zaroor zarur shayad kabhi hamesha pehle baad saath bina wajah taraf
tarah waise aise jaise isliye isiliye kyonki apna apni apne unka unki unke uska uski uske inka inki inke
kuchh sabse bada badi bade chota choti chote naya nayi naye purana purani sahi galat accha bura bure
hona hoga hogi honge hote hoti hota chahiye chahta chahti chahte lena dena diya di diye liya li
bata batao bataya batana samay waqt din raat subah shaam abhi jaldi der bahar andar upar neeche aage peeche
""".split())
HINGLISH_AMBIGUOUS = set("""
main mein me to the is us in un ho hum bas par kar ka ki ke ko se ya de do le lo na ne hi jo so
""".split())
EN_SURE_ZIPF = 3.5   # common English word
EN_RARE_ZIPF = 2.5   # rare in English: probably romanized Hindi (or a name)


def _in_ranges(cp: int, ranges) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _strip_edges(word: str) -> str:
    """Drop punctuation at both ends but keep internal apostrophes and hyphens."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", word)


def _script_counts(word: str, l1: str) -> tuple[int, int]:
    l1_n = lat_n = 0
    for ch in word:
        if not _is_letter(ch):
            continue
        cp = ord(ch)
        if _in_ranges(cp, SCRIPT_RANGES[l1]):
            l1_n += 1
        elif _in_ranges(cp, LATIN_RANGES):
            lat_n += 1
    return l1_n, lat_n


def tag_by_script(word: str, l1: str, l2: str = "en") -> str | None:
    """Return l1, l2, or None (no letters, e.g. a number or emoji)."""
    l1_n, lat_n = _script_counts(word, l1)
    if l1_n == 0 and lat_n == 0:
        return None
    return l1 if l1_n >= lat_n else l2


# ---------------------------------------------------------------- frequency method
@lru_cache(maxsize=None)
def _zipf(word: str, lang: str) -> float:
    from wordfreq import zipf_frequency  # imported lazily: slow to load, not needed for script pairs

    return zipf_frequency(word, lang)


def _has_latin(word: str) -> bool:
    return any(_is_letter(ch) and _in_ranges(ord(ch), LATIN_RANGES) for ch in word)


def _vietnamese_marks(word: str) -> bool:
    return any(0x1EA0 <= ord(ch) <= 0x1EFF for ch in word) or any(ch in DIACRITIC_HINTS["vi"] for ch in word)


def tag_by_frequency(word: str, l1: str, l2: str, prev: str | None) -> str | None:
    """Decide by diacritics, then word frequency, then the previous word's language."""
    core = _strip_edges(word).lower()
    if not core or not any(_is_letter(c) for c in core):
        return None
    if not _has_latin(core):
        return None  # e.g. emoji or a stray non-Latin symbol
    if l1 == "vi" and _vietnamese_marks(core):
        return l1
    hint1 = any(c in DIACRITIC_HINTS.get(l1, ()) for c in core)
    hint2 = any(c in DIACRITIC_HINTS.get(l2, ()) for c in core)
    if hint1 and not hint2:
        return l1
    if hint2 and not hint1:
        return l2
    z1, z2 = _zipf(core, l1), _zipf(core, l2)
    if z1 == 0.0 and z2 == 0.0:
        return prev or l1
    if z1 - z2 >= FREQ_MARGIN:
        return l1
    if z2 - z1 >= FREQ_MARGIN:
        return l2
    return prev or l1


# ---------------------------------------------------------------- romanized Hindi
def _tag_hinglish_one(word: str) -> str | None:
    """'hi', 'en', 'ambiguous', or None (no letters)."""
    by_script = tag_by_script(word, "hi", "en")
    if by_script is None:
        return None
    if by_script == "hi":  # Devanagari letters present
        return "hi"
    core = _strip_edges(word).lower()
    if not core:
        return None
    if core in HINGLISH_SURE:
        return "hi"
    if core in HINGLISH_AMBIGUOUS:
        return "ambiguous"
    z = _zipf(core, "en")
    if z >= EN_SURE_ZIPF:
        return "en"
    if z < EN_RARE_ZIPF:
        return "hi"
    return "ambiguous"


def tag_hinglish(words: list[str]) -> list[str | None]:
    """Two passes: sure words first, then ambiguous words take the majority of their neighbours."""
    first = [_tag_hinglish_one(w) for w in words]
    out: list[str | None] = list(first)
    for i, tag in enumerate(first):
        if tag != "ambiguous":
            continue
        neigh = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(first) and first[j] in ("hi", "en"):
                neigh.append(first[j])
        if not neigh:
            prev = next((out[j] for j in range(i - 1, -1, -1) if out[j] in ("hi", "en")), None)
            out[i] = prev or "hi"
        elif neigh.count("en") > neigh.count("hi"):
            out[i] = "en"
        elif neigh.count("hi") > neigh.count("en"):
            out[i] = "hi"
        else:
            out[i] = "hi"  # tie: these ambiguous forms are mostly Hindi function words (se, ka, ki, ko ...)
    return out


# ---------------------------------------------------------------- CJK segmentation
_SEGMENTERS: dict[str, object] = {}


def _segment(chunk: str, l1: str) -> list[str]:
    if l1 == "zh":
        if "zh" not in _SEGMENTERS:
            import jieba

            jieba.setLogLevel(60)
            _SEGMENTERS["zh"] = jieba
        return _SEGMENTERS["zh"].lcut(chunk)
    if l1 == "yue":
        if "yue" not in _SEGMENTERS:
            import pycantonese

            _SEGMENTERS["yue"] = pycantonese
        return _SEGMENTERS["yue"].segment(chunk)
    if l1 == "ja":
        if "ja" not in _SEGMENTERS:
            from fugashi import Tagger

            _SEGMENTERS["ja"] = Tagger()
        return [w.surface for w in _SEGMENTERS["ja"](chunk)]
    raise ValueError(l1)


def _has_cjk(chunk: str, l1: str) -> bool:
    return any(_in_ranges(ord(ch), SCRIPT_RANGES[l1]) for ch in chunk)


# ---------------------------------------------------------------- public entry point
def tokenize_and_tag(text: str, pair: str) -> list[tuple[str, str]]:
    """Split one sentence into words and tag each with a language code.

    Tokens without letters (numbers, bare punctuation, emoji) are dropped.
    The sticky rule for Latin-script pairs restarts at every sentence.
    """
    l1, _, l2, method = PAIRS[pair]
    out: list[tuple[str, str]] = []
    if method == "hinglish":
        words = text.split()
        return [(w, t) for w, t in zip(words, tag_hinglish(words)) if t is not None]
    prev: str | None = None
    for chunk in text.split():
        if method == "cjk" and _has_cjk(chunk, l1):
            pieces = [p for p in _segment(chunk, l1) if p.strip()]
        else:
            pieces = [chunk]
        for piece in pieces:
            if method == "freq":
                lang = tag_by_frequency(piece, l1, l2, prev)
            else:
                lang = tag_by_script(piece, l1, l2)
            if lang is None:
                continue
            out.append((piece, lang))
            prev = lang
    return out


if __name__ == "__main__":
    demo = [
        ("Spanish_eng", "Cuando termine mi maestría, I will invest in sustainable projects."),
        ("Arabic_eng", "أنا تابعت خبر الفيزا، honestly الموضوع معقد شوي."),
        ("Hindi_eng", "Woh kal Prime Minister se milne wali hain, to discuss the new tariffs."),
        ("Hindi_eng", "मुझे लगता है कि this project is interesting, लेकिन time कम है।"),
        ("Chinese_eng", "我今天下午有个meeting，你先帮我review一下那个文件。"),
        ("German_French", "Ich habe gestern un film français gesehen, c'était super."),
    ]
    for pair, sent in demo:
        print(pair)
        print("  ", " ".join(f"{w}/{l}" for w, l in tokenize_and_tag(sent, pair)))
