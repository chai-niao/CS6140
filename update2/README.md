# Update 2: Dual-head model, weighted loss, and the two-backbone experiment

Code and results for Project Update 2 of *Predictive Multitask Learning for Streaming
Code-Switching* (CS6140, Fall 2026). It builds on the data pipeline of `../update1/`
(language tagging, labels, 80/10/10 split, streaming windows) and adds the models, the
training loop, the evaluation and the figures.

## Contents

| File | Update 2 item | Description |
|---|---|---|
| `model.py` | 1 | `DualHeadModel`: a causal backbone plus a switch head (2 classes) and a duration head (3 bins). Model A = `xlm-roberta-base` with `config.is_decoder = True` (causal mask); Model B = `Qwen/Qwen2.5-0.5B` (causal by construction). Optional freezing of all but the top k layers |
| `losses.py` | 2 | `MultiTaskLoss`: `L = lambda_switch * CE_switch + lambda_duration * CE_duration`, both with `ignore_index = -1`; optional class weights for the 87/13 switch imbalance and the burst-length bins |
| `metrics.py` | 4 | Anticipatory F1 with per-pair thresholds tuned on dev, duration accuracy at true switches, F1 by switch type (intra, inter, tag), sigma across pairs |
| `engine.py` | | Device and mixed precision, batched prediction, per-split evaluation |
| `data.py` | | Loading the Update 1 conversations, subsets per pair, class weights |
| `train.py` | 2, 3, 5 | Training loop (AdamW, warmup + linear decay, AMP on CUDA, gradient clipping, DataParallel on two GPUs); logs both losses every 20 steps and the dev metrics periodically; final dev and test evaluation |
| `evaluate.py` | 3 | Re-evaluates a saved run and checks that word-by-word streaming inference gives the same P(switch) as the one-pass training window |
| `plots.py` | 4, 5 | Convergence curves (switch and duration loss, dev F1), per-pair F1 bar chart against the naive baseline, lambda-sweep figure and comparison tables |
| `kaggle/` | | `stage.py` prepares the two private Kaggle datasets (processed data, code) and one kernel per plan in `kaggle/plans/`; `run_kaggle_template.py` is the kernel script |
| `tests/` | | Unit tests: causal constraint of both backbones, `ignore_index` behaviour, lambda weighting, metrics, and an end-to-end smoke run with tiny random backbones |
| `results/` | | Figures and tables produced by `plots.py`, and `SUMMARY.md` with the headline numbers |

## Setup

Python 3.10 or newer. The Update 1 environment plus `matplotlib`:

```bash
cd update2
pip install -r ../update1/requirements.txt -r requirements.txt
pytest -q tests
```

`train.py` reads `../update1/data_processed/` (output of `python ../update1/split.py`), or any
directory given with `--data_dir`.

## Training and evaluation

```bash
python train.py --model xlmr --epochs 2 --batch_size 32                        # Model A, all 15 pairs, both GPUs
python train.py --model qwen --epochs 2 --batch_size 4 --grad_accum 8 --data_parallel 0   # Model B, one GPU
python train.py --model xlmr --max_train_convs 2000 --epochs 2 --lambda_duration 0.5   # a loss-weighting experiment
python evaluate.py --run_dir runs/<run> --streaming_check 5                    # test metrics + streaming consistency
python plots.py --runs "Model A (XLM-R)=runs/xlmr_full" "Model B (Qwen2.5-0.5B)=runs/qwen_full" \
                --sweep "0=runs/sweep_lam0" "0.5=runs/sweep_lam05" "1=runs/sweep_lam1" "2=runs/sweep_lam2"
```

Every run writes `config.json`, `train_log.csv` (both losses per logging step), `dev_log.csv`,
`dev_metrics.csv`, `test_metrics.csv`, `summary.json`, `results.md` and, unless `--save_model 0`,
the best checkpoint `model.pt` (fp16). The checkpoint is chosen by the dev mean F1 with per-pair tuned
thresholds, the metric we report (`--select_by f1_tuned`, default since 2026-11-10; the first full runs used
F1 at threshold 0.5, which picked the end of epoch 1 for XLM-R although the tuned F1 still rose).
`--select_by dur_acc` chooses by duration accuracy instead (used for the duration-only run).
`--amp auto` (default) uses bfloat16 on GPUs with native support (compute capability 8.0+) and
float16 with loss scaling elsewhere. On the T4, bfloat16 is emulated and about twice as slow
(`kaggle/plans/speed.json` measured 0.77 vs 0.41 s per step for XLM-R and 3.07 vs 1.40 s for Qwen).

The models were trained on Kaggle (2 x Tesla T4, free tier, at most two GPU sessions at a time).
`kaggle/stage.py` builds the uploads and one kernel per plan (`smoke`, `speed`, `lambda_sweep`,
`xlmr_full`, `qwen_full`); kernel slugs use dashes (`lambda_sweep` becomes `cs6140-update2-lambda-sweep`).
For example:

```bash
python kaggle/stage.py --user <kaggle user> --stage_dir /tmp/kaggle_stage --plan xlmr_full
kaggle datasets create -p /tmp/kaggle_stage/data      # once; later: kaggle datasets version -p ... -m "update"
kaggle datasets create -p /tmp/kaggle_stage/code
kaggle kernels push -p /tmp/kaggle_stage/kernel_xlmr_full
kaggle kernels output <kaggle user>/cs6140-update2-xlmr-full -p runs_from_kaggle/xlmr_full
```

## Design notes

* **Dual-head architecture.** One backbone, a shared LayerNorm + dropout, and two linear heads on the
  hidden state of every position. A position is scored only where a word ends (labels on the last subword), so the
  heads see the whole current word and nothing after it. The causal constraint is unit-tested
  for both backbones (`tests/test_model.py`) and demonstrated end to end by the streaming check
  of `evaluate.py`.
* **Loss weighting and `ignore_index = -1`.** `y_dur` is `-1` wherever there is no switch, so the
  duration cross-entropy is averaged over true switches only; `y_sw` is `-1` on non-final subwords,
  BOS, padding and positions already scored by an overlapping window. The switch class is
  up-weighted by `(1 - p) / p` (p = training switch rate, about 13%) by default, the duration
  bins by inverse frequency; both can be switched off (`--switch_weight none`,
  `--duration_weight none`). `lambda_switch` and `lambda_duration` are command-line arguments;
  the sweep in `kaggle/plans/lambda_sweep.json` compares `lambda_duration` in {0, 0.5, 1, 2},
  one run without class weights, and a duration-only run (`lambda_switch = 0`).
* **Experiment A: two backbones.** Same data, same heads, same loss, same evaluation; only the
  backbone changes. XLM-R base (278M parameters) is fine-tuned completely; for Qwen2.5-0.5B
  (494M) `--train_top_layers k` restricts training to the top k layers when memory is short.
* **Universality audit.** Thresholds are tuned per pair on dev (as for the naive baseline of
  Update 1) and frozen for test. `plots.py` draws the per-pair F1 of the baseline and of each
  model and reports the mean and sigma.
* **Convergence.** `train_log.csv` holds the switch and duration loss every 20 optimizer steps;
  `dev_log.csv` the dev loss, mean F1 at threshold 0.5, sigma and duration accuracy at every
  evaluation. `plots.py` turns them into `results/convergence_<run>.png`.
