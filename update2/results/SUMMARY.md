# Update 2 results

All numbers are on the test split (11,874 conversations, 15 language pairs) unless stated.
F1 is the anticipatory F1 of the switch class with a decision threshold tuned per pair on dev.
sigma is the population standard deviation of the 15 per-pair F1 values (smaller = more universal).
Training ran in Kaggle Notebooks on NVIDIA T4 GPUs in float16. No batch had a non-finite loss in any run.

## Experiment A: two backbones (all training data, 2 epochs, lambda_sw = lambda_dur = 1, class weights on)

| run | params | train min | mean F1 | sigma | F1 at 0.5 | duration acc | F1 tag | F1 inter | F1 intra |
|---|---|---|---|---|---|---|---|---|---|
| naive baseline (Update 1) | none | none | 0.644 | 0.068 | n/a | 0.617 | n/a | n/a | n/a |
| Model A: XLM-R base + causal mask | 277M | 41 (2 GPUs) | 0.784 | 0.070 | 0.714 | 0.694 | 0.842 | 0.833 | 0.682 |
| Model B: Qwen2.5-0.5B | 494M | 129 (1 GPU) | **0.805** | **0.064** | 0.745 | 0.704 | 0.859 | 0.853 | 0.710 |

* Both models beat the naive baseline clearly: +0.140 (A) and +0.161 (B) mean F1.
* Model B is ahead of Model A on all 15 pairs, on all three switch types and on duration accuracy, with a lower sigma.
  The lead is modest (0.021) and costs 1.8x the parameters, 3.2x the training time and 2x the peak GPU memory (10.2 vs 5.2 GB).
* Intra-sentential switches are the hardest type for both models.
* An earlier Model A run with 128-subword windows and 3 epochs (`run_logs/model_a_xlmr_window128_3ep`, before the duplicate
  filter) reached 0.788, so the 256-subword window did not lift F1.

Full table: [comparison.md](comparison.md). Per-pair numbers: [f1_by_pair.md](f1_by_pair.md).

## Universality audit

![Per-pair F1](f1_by_pair.png)

* Every pair improves. Model B gains from +0.112 (Hindi) to +0.226 (Cantonese) over the baseline.
* Best minus worst pair: 0.282 (baseline), 0.252 (A), 0.223 (B). Model B's lead over A is largest on the weakest pairs
  (Filipino +0.040, Cantonese +0.031).
* Filipino, Cantonese, Malay and Hindi stay the four weakest pairs for both models. They are the four pairs with the
  largest share of short switches (1 to 2 words: 36 to 52 percent of switches).

## Loss weighting (Model A, 2,000 training conversations per pair, 1 epoch, one seed)

| run | mean F1 | F1 at 0.5 | sigma | duration acc | final CE switch | final CE duration |
|---|---|---|---|---|---|---|
| lambda_dur = 0 (switch only) | **0.754** | 0.685 | 0.072 | n/a | 0.310 | n/a |
| lambda_dur = 0.5 | 0.745 | 0.683 | 0.071 | 0.659 | 0.321 | 0.815 |
| lambda_dur = 1 | 0.735 | 0.682 | 0.074 | 0.661 | 0.336 | 0.815 |
| lambda_dur = 2 | 0.730 | 0.674 | 0.077 | 0.664 | 0.345 | 0.810 |
| lambda_dur = 1, no class weights | 0.733 | 0.717 | 0.074 | 0.683 | 0.202 | 0.764 |
| duration only (lambda_sw = 0) | n/a | n/a | n/a | 0.657 | n/a | 0.839 |

![Loss weighting](lambda_sweep.png)

* With this budget the duration task costs switch F1: 0.754 alone, 0.730 at lambda_dur = 2. The switch task helps the
  duration head only slightly (0.657 alone, 0.659 to 0.664 together).
* Class weights do not change the tuned F1 (0.733 vs 0.735); they only shift P(switch) upward (tuned thresholds 0.75 to
  0.90 with weights, 0.25 to 0.55 without), which is why F1 at 0.5 is higher without them.
* Single runs on a subset; differences are at most 0.025. To be confirmed on all data with three seeds.

## Convergence

![Convergence of both heads](convergence_compare.png)

Per run, with dev F1 and duration accuracy: [Model A](convergence_model_a__xlm_r.png), [Model B](convergence_model_b__qwen2_5_0_5b.png).

* Both heads fall steeply in the first 0.2 epochs and level off; the duration head ends higher.
* Model B's training loss drops when the second epoch starts while its dev loss stays flat (0.944 to 0.958): it starts to
  memorize. Model A's dev loss still falls slowly (1.004 to 0.974).
* Dev F1 still rose slightly in epoch 2 (+0.008 A, +0.006 B); the last checkpoint was the best for both.
* Checkpoint selection: selecting by dev F1 at threshold 0.5 picked step 500 for Qwen in the first run
  (`run_logs/model_b_qwen_first_run_select_f1_at_0.5`, test F1 0.777). Selecting by the tuned F1 (now the default,
  `--select_by f1_tuned`) gives 0.805.

## Streaming check

Feeding a conversation one word at a time gives the same P(switch) as one pass over the window: largest difference
5.1e-7 (Model A) and 3.0e-6 (Model B), float32, 5 test conversations each (`run_logs/*/streaming_check.md`).

## float16 vs bfloat16 on the T4 (40-step check, `run_logs/speed_*`)

| model | bfloat16 | float16 + loss scaling |
|---|---|---|
| XLM-R | 0.77 s/step | 0.41 s/step |
| Qwen2.5-0.5B | 3.07 s/step | 1.40 s/step |

## Files

* `run_logs/<run>/`: `config.json`, `summary.json`, `train_log.csv` (every 20 steps), `dev_log.csv` (every 500 steps),
  `test_metrics.csv` and `dev_metrics.csv` (per pair), `streaming_check.md`, `train.log`.
* Model checkpoints (`model.pt`, 0.5 to 1 GB) are not in the repository.

## Code review fixes and local rerun (2026-11-11)

A review of the Update 2 code found ten issues. All are fixed on the branch `update2-review-fixes`; both models
were then retrained once on a single RTX 5070 Ti Laptop GPU (12 GB, bfloat16, seed 42, same data and schedule:
2 epochs, effective batch 32, 256-subword windows). Logs: `run_logs/model_a_xlmr_local`, `run_logs/model_b_qwen_local`;
figures and tables: `local_rerun/`.

| fix | what changed |
|---|---|
| gradient accumulation | each micro-batch loss is divided by the class-weighted target count of the whole optimizer step, so an accumulated step equals one large batch; before, every micro-batch counted equally, and length-bucketed micro-batches of short windows were over-weighted (Model B only, 4 x 8) |
| streaming check | 2 random conversations plus the longest one of every pair (45 in total) instead of the first 5 short conversations, which were all Arabic; compared with the real evaluation windows, including conversations that span several windows |
| streaming context | `iter_prefixes(..., max_len, stride)` limits the prefix to the training window that scores the word; XLM-R can no longer run past its 512 positions in a long conversation |
| baseline test set | `update1/baselines.py --exclude duplicates.json` scores the baseline on the same dev / test conversations as the models (`baseline_f1_dedup.csv`) |
| fp16 overflow | steps skipped by GradScaler are counted, logged and do not advance the LR schedule |
| `--duration_weight` | only `none` or `balanced`; other values raise an error instead of silently meaning `balanced` |
| best checkpoint | kept only in `model.pt` (no second copy in RAM) |
| host syncs | one per optimizer step instead of four per micro-batch |
| dev evaluation | the windows of the periodic dev subset are built once |
| shared metrics | F1 and threshold tuning live in `update1/common.py` and are used by the baseline and the models |

New option `--gradient_checkpointing 1`: Model B runs with micro-batches of 8 x 4 at 0.6 s per step and 9.3 GB
(4 x 8 without it needed 10.2 GB and spilled out of the 12 GB GPU).

| run | train min | mean F1 | sigma | F1 at 0.5 | duration acc | F1 tag | F1 inter | F1 intra | streaming check |
|---|---|---|---|---|---|---|---|---|---|
| naive baseline, duplicates excluded | none | 0.644 | 0.068 | n/a | 0.617 | n/a | n/a | n/a | n/a |
| Model A, Kaggle (2 x T4) | 41 | 0.784 | 0.070 | 0.714 | 0.694 | 0.842 | 0.833 | 0.682 | 5 Arabic, 5.1e-7 |
| Model A, local, fixed code | 10 | 0.784 | 0.070 | 0.713 | 0.694 | 0.843 | 0.834 | 0.680 | 45 (6 multi-window), 2.7e-6 |
| Model B, Kaggle (1 x T4) | 129 | 0.805 | 0.064 | 0.745 | 0.704 | 0.859 | 0.853 | 0.710 | 5 Arabic, 3.0e-6 |
| Model B, local, fixed code | 59 | 0.807 | 0.063 | 0.743 | 0.704 | 0.860 | 0.854 | 0.711 | 45 (10 multi-window), 5.7e-6 |

* Every conclusion above holds: Model B is ahead of Model A on all 15 pairs, the four weakest pairs are the same,
  and the convergence curves show the same epoch-2 memorization for Model B.
* The loss fix moves Model B by +0.002 mean F1, well inside seed noise; Model A is unchanged, as expected
  (it used no accumulation, so its loss was already the mean over the batch).
* Excluding the 8 duplicate test conversations changes no baseline F1 at three decimals (Cantonese duration
  accuracy 0.607 -> 0.606).
* Streaming and windowed predictions agree to 5.7e-6 or better on all 45 conversations, including the 16
  conversation-model cases that need more than one window.
