"""Predictive label generation (project spec, section II.B and II.C).

For every position t in a word stream with language tags lang[0..n-1]:

  y_sw[t]  = 1 if lang[t+1] != lang[t] else 0          (switch at the NEXT token)
  y_dur[t] = bin(length of the new-language run starting at t+1) if y_sw[t] == 1
           = -1 otherwise (ignored by the loss via ignore_index=-1)

Duration bins: 0 = Small (1 to 2 tokens), 1 = Medium (3 to 6), 2 = Large (7 or more).

The last position has no next token, so both labels are -1 there. A run that reaches
the end of the stream is counted up to the end (we cannot see how it would continue).
"""
from __future__ import annotations

IGNORE = -1
BIN_NAMES = {0: "Small (1-2)", 1: "Medium (3-6)", 2: "Large (7+)"}


def burst_bin(run_length: int) -> int:
    if run_length <= 2:
        return 0
    if run_length <= 6:
        return 1
    return 2


def make_labels(langs: list[str]) -> tuple[list[int], list[int]]:
    """Return (y_sw, y_dur), each the same length as `langs`."""
    n = len(langs)
    y_sw = [IGNORE] * n
    y_dur = [IGNORE] * n
    for t in range(n - 1):
        if langs[t + 1] != langs[t]:
            y_sw[t] = 1
            k = t + 1
            while k < n and langs[k] == langs[t + 1]:
                k += 1
            y_dur[t] = burst_bin(k - (t + 1))
        else:
            y_sw[t] = 0
    return y_sw, y_dur


def label_table(words: list[str], langs: list[str]) -> str:
    """Markdown table of (t, x_t, lang, x_{t+1}, y_sw, y_dur) for reports and sanity checks."""
    y_sw, y_dur = make_labels(langs)
    rows = ["| t | x_t | lang | x_{t+1} | y_sw(t) | y_dur(t) |", "|---|---|---|---|---|---|"]
    for t, (w, lg) in enumerate(zip(words, langs)):
        nxt = words[t + 1] if t + 1 < len(words) else ""
        dur = y_dur[t] if y_dur[t] == IGNORE else f"{y_dur[t]} ({BIN_NAMES[y_dur[t]].split()[0]})"
        rows.append(f"| {t + 1} | {w} | {lg} | {nxt} | {y_sw[t]} | {dur} |")
    return "\n".join(rows)


if __name__ == "__main__":
    words = "Cuando termine mi maestría, I will invest in sustainable projects.".split()
    langs = ["es"] * 4 + ["en"] * 6
    print(label_table(words, langs))
