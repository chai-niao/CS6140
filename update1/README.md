# Update 1: Streaming data pipeline and causal baselines

Code and results for Project Update 1 of *Predictive Multitask Learning for Streaming
Code-Switching* (CS6140, Fall 2026). Everything in this directory runs on a laptop CPU;
no model is trained at this stage.

## Contents

| File | Update 1 item | Description |
|---|---|---|
| `lid.py` | 2 | Word-level language identification for all 15 pairs: Unicode script (Arabic, Korean, Russian), segmenter + script (Mandarin, Cantonese, Japanese), word frequency + diacritics + neighbour rule (the 8 Latin-script pairs), and a Hinglish word list + English frequency for Hindi, 59% of which is romanized |
| `labels.py` | 2 | `y_sw(t)` and `y_dur(t)` for the upcoming token; duration bins 0/1/2; `-1` marks ignored positions |
| `split.py` | 1, 2 | Hugging Face dataset to tagged and labelled conversations; every pair is capped at 8,000 conversations; 80/10/10 split at the conversation level |
| `stats.py` | 1 | Per-pair statistics: switch rate, burst-length bins, share of the embedded language, breakdown by switch type |
| `baselines.py` | 4 | Naive baselines and Anticipatory F1 per pair, plus sigma across pairs |
| `dataset.py` | 2, 5 | Streaming DataLoader (PyTorch): prefix windows, subword alignment, labels on the last subword of each word |
| `causal_check.py` | 3 | Enables the causal mask for XLM-R, loads Qwen2.5-0.5B as the natively causal model, and runs a leak test |
| `audit.py` | 5 | Random sample of 300 tagged words with context for a manual accuracy check (`results/lid_audit_sample.csv`) |
| `run_all.sh` | | Runs the whole pipeline in order |
| `tests/` | | Unit tests for the label logic and the language tagger |
| `results/` | | Tables and logs produced by the scripts; `results/SUMMARY.md` collects the headline numbers |
| `data_processed/` | | Output of `split.py` (about 200 MB, not committed) |

## Setup

Python 3.10 or newer.

```bash
cd update1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests
```

The data is loaded directly from Hugging Face; nothing has to be downloaded by hand:

```python
from datasets import load_dataset
ds = load_dataset("Shelton1013/SwitchLingua_text")
```

The dataset is gated: request access on its Hugging Face page and run `hf auth login` once.
It comes as one `train` split of 234,172 rows; `split.py` assigns every row to its pair with
the `first_language` and `second_language` columns.

## Running the pipeline

```bash
./run_all.sh
```

or step by step:

```bash
python split.py                 # 1. tag, label and split all 15 pairs (about a minute)
python stats.py                 # 2. results/pair_stats.{csv,md}
python baselines.py             # 3. results/baseline_f1.{csv,md}
python dataset.py --demo        # 4. results/dataloader_demo.txt (downloads the two tokenizers)
python causal_check.py          # 5. results/leak_test.txt (downloads xlm-roberta-base and Qwen2.5-0.5B, about 1 GB each)
python audit.py                 # 6. results/lid_audit_sample.csv; the `correct` column is filled in by review
                                #    (an already reviewed sample is kept; --force draws a new one)
python audit.py --score         # 7. accuracy per method and per pair from the reviewed CSV
```

Other entry points:

```bash
python lid.py                   # tags a few example sentences
python labels.py                # label table for an example sentence
python split.py --pairs Spanish_eng Hindi_eng --max_convs 500   # quick partial run
```

## Design notes

* **Labels for the upcoming token.** `y_sw[t] = 1` iff `lang[t+1] != lang[t]`; `y_dur[t]` is the
  bin of the run length of the new language starting at `t+1`, and `-1` when there is no
  switch. The last position of a conversation has no target and is ignored.
* **Streaming DataLoader.** A conversation is one subword stream. Training uses windows of
  128 subwords with stride 64; because attention is causal, every position of a window is a
  valid prefix, so one forward pass scores all prefixes. `iter_prefixes()` provides the true
  word-by-word view used at inference time.
* **Subword alignment.** Every word is tokenized on its own, so no subword spans two words
  (the natural tokenization of Chinese and Japanese merges words, e.g. 帮+我 into one token,
  which would let position t see part of word t+1). `split.py` stores `space_before` for
  every word and `dataset.py` restores the original spacing: no spaces are inserted between
  Chinese, Cantonese or Japanese words, and the decoded subwords reproduce the original text.
  Each subword inherits the language of its word. The prediction labels sit on the last
  subword of a word, which is the first position that has seen the whole word; the other
  subwords get `-1`.
* **Same input for training and streaming.** Training windows and `iter_prefixes()` both
  start with the BOS token (`<s>` for XLM-R; Qwen2.5 has none); `tests/test_dataset.py`
  checks that every streaming prefix is a prefix of the training window.
* **Causal constraint.** `config.is_decoder = True` makes `XLMRobertaModel` apply a causal
  mask; the leak test verifies it on one sentence and on a real padded DataLoader batch
  from four pairs. Qwen2.5-0.5B is causal by construction.
* **Naive baselines.** `never` (F1 = 0), `always` (F1 = 2p/(1+p)), `lang` (P(switch | language
  of x_t)) and `token` (P(switch | x_t), decision threshold tuned on the dev split). The token
  lookup is the model that predicts a switch from the current token only.
* **Punctuation.** Latin-script words keep attached punctuation ("maestría,"); in
  Mandarin, Cantonese and Japanese the segmenter's punctuation pieces ("，", "。") are
  attached to the previous word in the same way instead of being dropped, so the
  punctuation cue is available in every pair.
* **Romanized Hindi.** Hindi words that are also frequent English words (mile, log, mere,
  tab, din, ...) are treated as ambiguous and take the language of their neighbours.
* **Known limitations of the tagger.** Romanized Hindi, proper names and loanwords in the
  Latin-script pairs, and segmentation errors in Chinese and Japanese. See the audit sample
  in `results/` and section 5 of `results/SUMMARY.md`.
