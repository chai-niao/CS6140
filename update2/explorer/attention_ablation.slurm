#!/bin/bash
# CS6140 Update 2, response to the Update 1 feedback: train XLM-R with or without the causal mask,
# then test it with the causal mask. Same settings as Model A (results/run_logs/model_a_xlmr_local).
#   sbatch --export=ALL,ATTN=bidirectional attention_ablation.slurm
#   sbatch --export=ALL,ATTN=causal        attention_ablation.slurm   (same-hardware control)
#SBATCH --job-name=cs6140-attn
#SBATCH --account=s.sushmita   # GPU allocation used for the reported runs; use your own
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
module load python/3.13.5
ATTN=${ATTN:?set ATTN=bidirectional or ATTN=causal}
cd "$HOME/cs6140/CS6140_repo/update2"
PY="$HOME/cs6140/venv/bin/python"
RUN="runs/xlmr_train_${ATTN}"
echo "host $(hostname)  job ${SLURM_JOB_ID}  attention ${ATTN}"; nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" -c "import torch,sys; print('torch', torch.__version__, 'cuda ok', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
"$PY" train.py --model xlmr --attention "$ATTN" --epochs 2 --batch_size 32 --max_len 256 --stride 128 \
    --lambda_switch 1 --lambda_duration 1 --select_by f1_tuned --seed 42 \
    --run_name "XLM-R trained ${ATTN}" --out_dir "$RUN"
# test with the causal mask (for ATTN=causal this is the run itself) and with the bidirectional mask,
# thresholds tuned again on dev under each test-time attention
"$PY" evaluate.py --run_dir "$RUN" --attention causal --streaming_check 2
"$PY" evaluate.py --run_dir "$RUN" --attention bidirectional
echo done
