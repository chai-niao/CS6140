# Update 1 results

Produced on 2026-10-12 by the scripts in `update1/` (see the README). The numbers below are
taken from `pair_stats.md`, `baseline_f1.md`, `dataloader_demo.txt` and `leak_test.txt`.

## 1. Language pairs

All 15 pairs of SwitchLingua_text: 14 with English (Arabic, Cantonese, Mandarin, Hindi,
Japanese, Korean, Russian, Spanish, French, German, Italian, Malay, Filipino, Vietnamese)
plus German with French. Every pair is capped at 8,000 conversations (Cantonese subsampled
from 119,248) and split 80/10/10 at the conversation level. Total: 118,798 conversations,
6.68 M words.

Rationale for testing universality: seven writing systems (Latin, Arabic, Devanagari,
Cyrillic, Hangul, Han, kana), languages with and without spaces between words, SOV and
SVO word orders, and one pair without English (German with French).

## 2. Data engineering

* The dataset provides no word-level language tags. They are derived with three methods:
  script (Arabic, Korean, Russian), segmenter + script (Mandarin, Cantonese, Japanese),
  word frequency + diacritics + neighbour rule (the 8 Latin-script pairs), and a hybrid
  for Hindi because 59% of the Hindi conversations are romanized ("Woh kal ... milne wali hain").
* Labels: `y_sw(t) = [lang(x_{t+1}) != lang(x_t)]`; `y_dur(t)` is the bin of the run length
  of the new language starting at t+1, and -1 otherwise. See `label_example.md`.
* Switch rate per word position: mean 13.9% across pairs (min 8.7% Japanese, max 24.5% Hindi).
  Between 86% and 91% of positions are therefore "no switch", which motivates class
  weighting and F1 as the metric.
* All three burst-length bins are well populated (Small 22 to 53%, Medium 11 to 29%,
  Large 18 to 65%).
* Streaming DataLoader: windows of 128 subwords with stride 64; labels on the last subword
  of each word; `iter_prefixes()` gives the word-by-word view. On the Spanish demo
  conversation XLM-R produces 1.64 subwords per word and Qwen2.5 1.52.

## 3. Causal constraint

Leak test on "Cuando termine mi maestría, | I will invest in sustainable projects.":
all tokens after "maestría," are replaced by random ids (3 trials) and the hidden states
at positions up to and including "maestría," are compared with the original run.

| model | max change at positions <= t | verdict |
|---|---|---|
| XLM-R without mask (reference) | 1.73 | leaks: the prefix representation depends on the future |
| Model A: XLM-R + causal mask (`config.is_decoder = True`) | 0.00 | no leak |
| Model B: Qwen2.5-0.5B | 0.00 | no leak (causal by construction) |

## 4. Baseline metrics: Anticipatory F1 on the test split

| baseline | mean F1 over 15 pairs | sigma |
|---|---|---|
| never switch | 0.000 | 0 |
| always switch | 0.244 | |
| P(switch given language of x_t) | 0.253 | |
| P(switch given x_t), punctuation stripped | 0.489 | 0.070 |
| **P(switch given x_t), token as is (naive model)** | **0.637** | **0.053** |

Per pair, the token lookup ranges from 0.524 (Mandarin) to 0.747 (Korean). About 0.15 F1
of its score comes from trailing punctuation: a comma or period on the current word is a
strong cue that the next clause may switch. Duration accuracy at true switches: majority
bin 0.509, bin conditioned on the current token 0.608.

The naive baseline is therefore a meaningful reference point: the transformer models have
to improve on F1 = 0.64 and sigma = 0.05 rather than on zero.

## 5. Challenges and next steps

* Romanized Hindi: script rules do not apply, so a Hinglish word list plus English
  frequency is used; proper names are still mis-tagged, because a name unknown to the
  English frequency list looks like Hindi.
* Latin-script pairs: names and loanwords ("fans", "meeting") are ambiguous; the neighbour
  rule helps, and a manual audit (`lid_audit_sample.csv`, 300 words) is used to report
  tagging accuracy.
* Cantonese and Hindi show the highest switch rates (21%, 24%) with many 1 to 2 word bursts;
  part of that is tagger noise rather than real switching.
* Subword alignment: labels must sit on the last subword of a word; the two tokenizers cut
  words differently (XLM-R SentencePiece vs Qwen byte-level BPE), and Qwen splits trailing
  punctuation into its own token with a leading space when words are pre-split
  ("Yemen." -> "ĠYemen", "Ġ."). Offset-mapping alignment for Qwen is planned.
* Next: train Model A (weeks 7 to 8) and Model B (weeks 9 to 10) on this pipeline.
