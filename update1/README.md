# Update 1: Streaming data pipeline and causal baselines

Code and results for Project Update 1 of *Predictive Multitask Learning for Streaming
Code-Switching* (CS6140, Fall 2026). Everything in this directory runs on a laptop CPU;
no model is trained at this stage.

## Contents

| File | Update 1 item | Description |
|---|---|---|
| `lid.py` | 2 | Word-level language identification for all 15 pairs: Unicode script (Arabic, Korean, Russian), segmenter + script (Mandarin, Cantonese, Japanese), word frequency + diacritics + neighbour rule (the 8 Latin-script pairs), and a Hinglish word list + English frequency for Hindi, 59% of which is romanized |
| `labels.py` | 2 | `y_sw(t)` and `y_dur(t)` for the upcoming token; duration bins 0/1/2; `-1` marks ignored positions |
| `split.py` | 1, 2 | Raw CSVs to tagged and labelled conversations; every pair is capped at 8,000 conversations; 80/10/10 split at the conversation level |
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

The raw CSVs of `Shelton1013/SwitchLingua_text` (gated on Hugging Face) are expected in
`../data/SwitchLingua_text/`. Another location can be given with the environment variable
`SWITCHLINGUA_DIR`.

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
python audit.py                 # 6. results/lid_audit_sample.csv; the `correct` column is filled in manually
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
* **Subword alignment.** Each subword inherits the language of its word. The prediction
  labels sit on the last subword of a word, which is the first position that has seen the
  whole word; the other subwords get `-1`.
* **Causal constraint.** `config.is_decoder = True` makes `XLMRobertaModel` apply a causal
  mask; the leak test verifies it. Qwen2.5-0.5B is causal by construction.
* **Naive baselines.** `never` (F1 = 0), `always` (F1 = 2p/(1+p)), `lang` (P(switch | language
  of x_t)) and `token` (P(switch | x_t), decision threshold tuned on the dev split). The token
  lookup is the model that predicts a switch from the current token only.
* **Known limitations of the tagger.** Romanized Hindi, proper names and loanwords in the
  Latin-script pairs, and segmentation errors in Chinese and Japanese. See the audit sample
  in `results/` and section 5 of `results/SUMMARY.md`.
