# Attention ablation: train without the causal mask, test with it (Update 1 feedback)

Question from the instructor at Update 1: what happens to a bidirectional model that is trained without the
causal mask and tested with it?

XLM-R base, the configuration of Model A (all data, 2 epochs, batch 32, 256-subword windows with stride 128,
lambda_sw = lambda_dur = 1, class weights, seed 42). Only the attention mask changes. Both runs trained on one
NVIDIA V100-SXM2-32GB (Northeastern Explorer cluster, float16 with loss scaling, `explorer/attention_ablation.slurm`;
torch 2.14.1 built for CUDA 12.6, since the default CUDA 13 wheel does not run on the cluster's driver).
Thresholds are tuned per pair on dev under the test-time attention and frozen for test. Test split, 15 pairs.

| trained with | tested with | mean F1 | sigma | F1 at 0.5 | duration acc | F1 intra | streaming model? |
|---|---|---|---|---|---|---|---|
| naive baseline | | 0.644 | 0.068 | n/a | 0.617 | n/a | yes |
| no mask (bidirectional) | causal mask | 0.615 | 0.087 | 0.517 | 0.311 | 0.483 | yes |
| causal mask | causal mask | 0.784 | 0.069 | 0.714 | 0.695 | 0.680 | yes |
| causal mask | no mask | 0.813 | 0.059 | 0.750 | 0.680 | 0.731 | no |
| no mask (bidirectional) | no mask | 0.972 | 0.024 | 0.961 | 0.988 | 0.972 | no |

* Trained without the mask and tested with it, XLM-R scores 0.615: below the naive word-lookup baseline
  (0.644), and below it on 8 of 15 pairs. Duration accuracy 0.311 is below always predicting the most
  frequent bin (0.501). Intra-sentential switches suffer most (0.483 against 0.680).
* Without the mask at test time the model reads the words after position t: 0.972 F1 and 0.988
  duration accuracy for the bidirectional model, 0.813 even for the causally trained one. These numbers are not
  reachable by a system that receives words one at a time.
* The causal-causal cell reproduces Model A (0.784, against 0.784 on the RTX 5070 Ti and on Kaggle).
  Evaluating the Kaggle Model A checkpoint without the mask on an Apple M4 gives 0.816, in line with 0.813.
* Streaming check of the bidirectionally trained model under the causal mask: word by word equals one pass
  (45 conversations, `run_logs/attention_train_bidirectional_explorer/streaming_check_causal.md`).

Commands:

```bash
python train.py --model xlmr --attention bidirectional --epochs 2 --batch_size 32 --out_dir runs/xlmr_train_bidirectional
python evaluate.py --run_dir runs/xlmr_train_bidirectional --attention causal --streaming_check 2
python train.py --model xlmr --attention causal --epochs 2 --batch_size 32 --out_dir runs/xlmr_train_causal
python evaluate.py --run_dir runs/xlmr_train_causal --attention bidirectional
```
