# Update 1 results

Produced on 2026-10-12 by the scripts in `update1/` (see the README); re-run on 2026-10-13
after the fixes listed in section 6. The numbers below are
taken from `pair_stats.md`, `baseline_f1.md`, `dataloader_demo.txt` and `leak_test.txt`.

## 1. Language pairs

All 15 pairs of SwitchLingua_text: 14 with English (Arabic, Cantonese, Mandarin, Hindi,
Japanese, Korean, Russian, Spanish, French, German, Italian, Malay, Filipino, Vietnamese)
plus German with French. Every pair is capped at 8,000 conversations (Cantonese subsampled
from 119,248) and split 80/10/10 at the conversation level. Total: 118,798 conversations,
6.64 M words.

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
* Switch rate per word position: mean 13.3% across pairs (min 8.7% Japanese, max 24.1% Hindi).
  Between 76% and 91% of positions are therefore "no switch", which motivates class
  weighting and F1 as the metric.
* All three burst-length bins are well populated (Small 22 to 52%, Medium 11 to 29%,
  Large 18 to 65%).
* Streaming DataLoader: windows of 128 subwords with stride 64; labels on the last subword
  of each word; `iter_prefixes()` gives the word-by-word view and starts with the same BOS
  token as the training windows. Every word is tokenized on its own with its original
  spacing, so no subword spans two words and Chinese and Japanese text is not padded with
  spaces. On the Spanish demo conversation XLM-R produces 1.64 subwords per word and
  Qwen2.5 1.40.

## 3. Causal constraint

Leak test on "Cuando termine mi maestría, | I will invest in sustainable projects.":
all tokens after "maestría," are replaced by random ids (3 trials) and the hidden states
at positions up to and including "maestría," are compared with the original run.

| model | max change at positions <= t | verdict |
|---|---|---|
| XLM-R without mask (reference) | 1.59 | leaks: the prefix representation depends on the future |
| Model A: XLM-R + causal mask (`config.is_decoder = True`) | 0.00 | no leak |
| Model B: Qwen2.5-0.5B | 0.00 | no leak (causal by construction) |

The same test on a real batch from the training DataLoader (8 dev conversations from
Spanish, Mandarin, Japanese and Hindi; BOS token, right padding, 24 to 129 subwords per row;
a random labelled position t in every row):

| model | max change at positions <= t | row alone vs in padded batch | verdict |
|---|---|---|---|
| XLM-R without mask (reference) | 2.43 | 6e-06 | leaks |
| Model A: XLM-R + causal mask | 0.00 | 3e-06 | no leak, padding has no effect |
| Model B: Qwen2.5-0.5B | 0.00 | 0.00 | no leak, padding has no effect |

## 4. Baseline metrics: Anticipatory F1 on the test split

| baseline | mean F1 over 15 pairs | sigma |
|---|---|---|
| never switch | 0.000 | 0 |
| always switch | 0.232 | |
| P(switch given language of x_t) | 0.245 | |
| P(switch given x_t), punctuation stripped | 0.475 | 0.060 |
| **P(switch given x_t), token as is (naive model)** | **0.644** | **0.068** |

Per pair, the token lookup ranges from 0.486 (Cantonese) to 0.768 (Japanese). About 0.17 F1
of its score comes from trailing punctuation: a comma or period on the current word is a
strong cue that the next clause may switch. Duration accuracy at true switches: majority
bin 0.501, bin conditioned on the current token 0.617.

The naive baseline is therefore a meaningful reference point: the transformer models have
to improve on F1 = 0.64 and sigma = 0.07 rather than on zero.

## 5. Challenges and next steps

* Romanized Hindi: script rules do not apply, so a Hinglish word list plus English
  frequency is used; proper names are still mis-tagged, because a name unknown to the
  English frequency list looks like Hindi.
* Tagging accuracy, audit of 300 sampled words (drawn before the fixes in section 6; the
  sampled positions are unchanged, but Hindi tags and Chinese, Cantonese and Japanese
  word boundaries have changed since) (`lid_audit_sample.csv`, 20 per pair, each
  judged in context; the review was done with LLM assistance and spot-checked by the team):
  298 of 300 correct (99.3%). By method: script 60/60, segmenter + script 60/60, frequency
  159/160, Hinglish 19/20. The two errors are the Hindi postposition "ko" tagged English
  next to the word "technology", and the English "era" in "a new era." tagged Spanish,
  where "era" is also a frequent Spanish word. Four further rows are borderline (the name
  "Kevon" in Cantonese text, the mixed token "referee" + Korean particle, "Coach" in
  "Coach Kerr"); counting them as errors gives 98.0%. Names and loanwords therefore remain
  the main source of noise, as expected.
* Hindi (24%) and Filipino (20%) show the highest switch rates with many 1 to 2 word bursts;
  part of that is tagger noise in romanized Hindi and Taglish loanwords rather than real
  switching. Cantonese, previously 21%, was inflated by a segmenter bug (section 6) and is
  12.8% after the fix. Romanized Cantonese particles (la, lor, wor) are not a problem: they
  account for 0.2% of switches into English.
* Subword alignment: labels must sit on the last subword of a word; the two tokenizers cut
  words differently (XLM-R SentencePiece vs Qwen byte-level BPE). Tokenizing the whole
  sentence and aligning by offsets would be simpler, but both tokenizers merge Chinese and
  Japanese words into one token (帮+我, があるので), which would leak part of word t+1 into
  position t. Words are therefore tokenized one at a time with their original spacing.
* Next: train Model A (weeks 7 to 8) and Model B (weeks 9 to 10) on this pipeline.

## 6. Fixes after code review (2026-10-13)

| problem | fix | effect on the numbers |
|---|---|---|
| pycantonese 5.0.0 gives every unseen ASCII token a new character `chr(0xE000 + n)`; after about 7,900 new English words `n` runs into full-width punctuation, and every later "，" came back as "Kevon" (other full-width punctuation was hit the same way: the frequent fake words "ATACMS" and "Hand" also disappear after the fix). 5,031 of 8,000 Cantonese conversations were affected. | Latin words, numbers and hyphens are cut out of CJK chunks before segmentation, so the segmenter only sees CJK text (`lid._segment_mixed`); regression test checks that its mapping no longer grows | Cantonese switch rate 21.4% -> 12.8%, token-lookup F1 0.649 -> 0.486 (now the hardest pair) |
| Chinese, Cantonese and Japanese punctuation ("，", "。") was dropped by the tagger, so these pairs lost the punctuation cue that other pairs keep on the word | punctuation pieces are attached to the previous word, as in "maestría," | F1 Mandarin 0.524 -> 0.658, Japanese 0.625 -> 0.768 |
| `is_split_into_words` put a space before every Chinese and Japanese word (`Ġ我 Ġ今天`) and before split-off punctuation (`ĠYemen Ġ.`) | `split.py` stores `space_before`; `dataset.encode_words` tokenizes each word alone with its original spacing (decoded subwords reproduce the text) | Qwen subwords per word on the Spanish demo 1.52 -> 1.40 |
| training windows started with `<s>` for XLM-R but `iter_prefixes()` and the leak test did not | one `bos_id()` used by all three; a test checks that each streaming prefix is a prefix of the training window | none on the baselines |
| Hinglish list marked frequent English words (mile, log, mere, tab, din, fir, jab, lag) as always Hindi, so "a mile to log in" became Hindi | moved to the ambiguous list (neighbour rule); added pe, ab, mai, hu, maine | Hindi switch rate 24.5% -> 24.1% |
| `run_all.sh` ends with `audit.py`, which overwrote the reviewed audit sample | `audit.py` keeps a reviewed sample unless `--force` | none |

Mean token-lookup F1 over the 15 pairs moves from 0.637 to 0.644 and sigma from 0.053 to 0.068.
