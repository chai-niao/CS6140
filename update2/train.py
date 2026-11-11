"""Train the dual-head model on the streaming windows of Update 1 and evaluate it (Update 2).

    python train.py --model xlmr                       # Model A: XLM-R base + causal mask, all 15 pairs
    python train.py --model qwen --train_top_layers 8  # Model B: Qwen2.5-0.5B, only the top 8 layers trainable
    python train.py --model xlmr --lambda_duration 0.5 --max_train_convs 2000   # a lambda experiment on a subset

Writes to --out_dir:
    config.json        all arguments, device, parameter counts
    train.log          everything printed
    train_log.csv      step, epoch, lr, total / switch / duration loss, seconds per step   (convergence curves)
    dev_log.csv        periodic dev evaluation: losses, mean F1 at threshold 0.5, sigma, duration accuracy
    dev_metrics.csv    final per-pair dev metrics; the per-pair thresholds tuned here are frozen for the test split
    test_metrics.csv   final per-pair test metrics (Anticipatory F1, duration accuracy, F1 by switch type)
    summary.json       headline numbers (mean F1, sigma, duration accuracy, ...)
    results.md         the test table and summary as Markdown
    model.pt           best checkpoint by dev mean F1 with tuned thresholds (--select_by), fp16; skipped with --save_model 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import setup_path  # noqa: F401
from common import markdown_table
from data import (DEFAULT_DATA_DIR, available_pairs, duplicate_ids, duration_class_weight, duration_counts, load_convs,
                  switch_class_weight, switch_statistics)
from batching import make_train_loader
from dataset import get_tokenizer
from engine import AutocastForward, amp_dtype_for, autocast, evaluate_convs, get_device, to_device
from losses import MultiTaskLoss
from model import DualHeadModel, save_model


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="xlmr", help="xlmr | qwen | any Hugging Face model name")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--pairs", nargs="*", default=None, help="default: every pair found in data_dir")
    ap.add_argument("--max_train_convs", type=int, default=0, help="training conversations per pair (0 = all)")
    ap.add_argument("--max_dev_convs", type=int, default=100,
                    help="dev conversations per pair for the periodic evaluation; the final dev evaluation uses all of them")
    ap.add_argument("--max_test_convs", type=int, default=0, help="test conversations per pair (0 = all)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-5, help="learning rate of the backbone")
    ap.add_argument("--head_lr", type=float, default=None, help="learning rate of the two heads (default 5 x lr)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda_switch", type=float, default=1.0)
    ap.add_argument("--select_by", default="f1_tuned", choices=["f1_tuned", "switch_f1", "dur_acc"],
                    help="dev metric that picks the best checkpoint: mean F1 with per-pair tuned thresholds (the reported "
                         "metric), mean F1 at threshold 0.5, or duration accuracy (for duration-only runs)")
    ap.add_argument("--lambda_duration", type=float, default=1.0)
    ap.add_argument("--switch_weight", default="balanced", help="class weight of the switch label: none | balanced | <number>")
    ap.add_argument("--duration_weight", default="balanced", help="class weights of the duration bins: none | balanced")
    ap.add_argument("--train_top_layers", type=int, default=0,
                    help="0 = fine-tune the whole backbone; k = freeze all but its top k layers (heads always train)")
    ap.add_argument("--max_len", type=int, default=256, help="window length in subwords (256 since Update 2)")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--bucket_batches", type=int, default=50,
                    help="sort windows by length within chunks of this many batches (0 = plain random batches)")
    ap.add_argument("--amp", default="auto", help="mixed precision on CUDA: auto | bf16 | fp16 | 0")
    ap.add_argument("--data_parallel", type=int, default=1, help="use every visible GPU (1 / 0)")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--eval_every", type=int, default=500, help="optimizer steps between dev evaluations (0 = epoch ends only)")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--time_budget_min", type=float, default=0,
                    help="stop training after this many minutes and go on to the final evaluation (0 = no limit)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--save_model", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiny", action="store_true", help="random two-layer backbone with the real tokenizer (tests)")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--out_dir", default=None, help="default: runs/<run_name or model>_<timestamp>")
    return ap.parse_args(argv)


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model: DualHeadModel, lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with a higher learning rate for the randomly initialised heads; no weight decay on biases and norms."""
    groups = []
    for module, group_lr in ((model.backbone, lr), (nn.ModuleList([model.head_norm, model.switch_head, model.duration_head]), head_lr)):
        decay = [p for n, p in module.named_parameters() if p.requires_grad and p.ndim > 1]
        no_decay = [p for n, p in module.named_parameters() if p.requires_grad and p.ndim <= 1]
        if decay:
            groups.append({"params": decay, "lr": group_lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": group_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def linear_schedule(optimizer, warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    amp_dtype = amp_dtype_for(device, args.amp)
    head_lr = args.head_lr if args.head_lr is not None else 5 * args.lr

    out_dir = Path(args.out_dir or f"runs/{args.run_name or args.model}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train.log")
    log(f"== train.py  {time.strftime('%Y-%m-%d %H:%M:%S')}  out_dir = {out_dir}")
    log(f"device {device}, mixed precision {amp_dtype}, torch {torch.__version__}")

    # ------------------------------------------------------------------ data
    tok = get_tokenizer(args.model)
    pairs = args.pairs or available_pairs(args.data_dir)
    train_convs = load_convs(args.data_dir, "train", pairs, args.max_train_convs, args.seed)
    dev_full = load_convs(args.data_dir, "dev", pairs)
    dev_sub = load_convs(args.data_dir, "dev", pairs, args.max_dev_convs, args.seed) if args.max_dev_convs else dev_full
    test_convs = load_convs(args.data_dir, "test", pairs, args.max_test_convs, args.seed)
    n_pos, n_sw = switch_statistics(train_convs)
    log(f"{len(pairs)} pairs: {' '.join(pairs)}")
    log(f"conversations: train {len(train_convs):,}, dev {len(dev_full):,} (periodic eval on {len(dev_sub):,}), test {len(test_convs):,}; "
        f"dev and test exclude {len(duplicate_ids('dev'))} and {len(duplicate_ids('test'))} conversations that duplicate an earlier split")
    log(f"training positions {n_pos:,}, switches {n_sw:,} ({100 * n_sw / max(n_pos, 1):.1f}%), duration bins {duration_counts(train_convs)}")

    t0 = time.perf_counter()
    train_loader = make_train_loader(train_convs, tok, batch_size=args.batch_size, max_len=args.max_len, stride=args.stride,
                                     bucket_batches=args.bucket_batches, seed=args.seed)
    log(f"{len(train_loader.dataset):,} training windows of <= {args.max_len} subwords, {len(train_loader):,} batches per epoch "
        f"(tokenized in {time.perf_counter() - t0:.0f}s)")

    # ------------------------------------------------------------------ model and loss
    sw_weight = switch_class_weight(train_convs, args.switch_weight)
    dur_weight = duration_class_weight(train_convs, args.duration_weight)
    loss_fn = MultiTaskLoss(args.lambda_switch, args.lambda_duration, sw_weight, dur_weight).to(device)
    model = DualHeadModel(args.model, dropout=args.dropout, tiny_vocab=len(tok) if args.tiny else None)
    if args.train_top_layers > 0:
        model.train_top_layers_only(args.train_top_layers)
    model.to(device)
    log(model.describe())
    log(loss_fn.describe())
    runner: nn.Module = model
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1 and args.data_parallel:
        runner = nn.DataParallel(AutocastForward(model, device.type, amp_dtype))
        log(f"DataParallel over {n_gpus} GPUs")

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(args.warmup_ratio * total_steps)
    optimizer = make_optimizer(model, args.lr, head_lr, args.weight_decay)
    scheduler = linear_schedule(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"{total_steps:,} optimizer steps ({args.epochs} epochs x {steps_per_epoch:,}), warmup {warmup}, lr {args.lr:g}, head lr {head_lr:g}, "
        f"batch {args.batch_size} x accumulation {args.grad_accum}")

    total, n_trainable = model.count_parameters()
    (out_dir / "config.json").write_text(json.dumps({
        **vars(args), "head_lr": head_lr, "device": str(device), "amp_dtype": str(amp_dtype), "n_gpus": n_gpus,
        "tokenizer": tok.name_or_path, "parameters": total, "trainable_parameters": n_trainable,
        "train_conversations": len(train_convs), "train_windows": len(train_loader.dataset), "total_steps": total_steps,
        "switch_class_weight": sw_weight, "duration_class_weight": dur_weight}, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ dev evaluation during training
    best = {"score": -1.0, "step": 0, "state": None}
    t_start = time.perf_counter()

    def dev_eval(step: int, epoch: float) -> None:
        res = evaluate_convs(runner, dev_sub, tok, device, amp_dtype, loss_fn, batch_size=args.eval_batch_size,
                             max_len=args.max_len, stride=args.stride)
        f1 = np.array([r["F1_at_0.5"] for r in res["rows"]])
        row = {"step": step, "epoch": round(epoch, 3), **{f"dev_loss_{k}": round(v, 4) for k, v in res["losses"].items()},
               "dev_mean_F1_at_0.5": round(float(f1.mean()), 4), "dev_sigma_F1_at_0.5": round(float(f1.std(ddof=0)), 4),
               "dev_mean_F1_tuned": res["summary"]["mean_F1"], "dev_mean_dur_acc": res["summary"]["mean_dur_acc"],
               "elapsed_min": round((time.perf_counter() - t_start) / 60, 1)}
        append_row(out_dir / "dev_log.csv", row)
        log(f"  dev @ step {step}: loss {row.get('dev_loss_total', float('nan')):.4f} "
            f"(switch {row.get('dev_loss_switch', float('nan')):.4f}, duration {row.get('dev_loss_duration', float('nan')):.4f}), "
            f"mean F1@0.5 {row['dev_mean_F1_at_0.5']:.4f}, sigma {row['dev_sigma_F1_at_0.5']:.4f}, dur acc {row['dev_mean_dur_acc']:.4f}")
        score = {"f1_tuned": row["dev_mean_F1_tuned"], "switch_f1": row["dev_mean_F1_at_0.5"],
                 "dur_acc": row["dev_mean_dur_acc"]}[args.select_by]
        if score > best["score"]:
            best.update(score=score, step=step,
                        state={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
            if args.save_model:
                save_model(model, out_dir / "model.pt", {"step": step, "select_by": args.select_by, "dev_score": score})
        runner.train()

    # ------------------------------------------------------------------ training loop
    step = 0
    done = False
    running = {"total": 0.0, "switch": 0.0, "duration": 0.0, "n": 0}
    nonfinite = 0
    stopped_by_budget = False
    t_log = time.perf_counter()
    for epoch in range(args.epochs):
        runner.train()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            b = to_device(batch, device)
            with autocast(device, amp_dtype):
                sw_logits, dur_logits = runner(b["input_ids"], b["attention_mask"])
            losses = loss_fn(sw_logits, dur_logits, b["y_sw"], b["y_dur"])
            loss = losses["total"] / args.grad_accum
            if torch.isfinite(loss.detach()):
                (scaler.scale(loss) if scaler else loss).backward()
                for k in ("total", "switch", "duration"):
                    running[k] += float(losses[k].detach())
                running["n"] += 1
            else:  # fp16 overflow in the forward pass: drop this batch instead of corrupting the weights
                nonfinite += 1
                if nonfinite <= 5 or nonfinite % 100 == 0:
                    log(f"non-finite loss at step {step} (batch {i}); skipped, {nonfinite} so far")
            if (i + 1) % args.grad_accum and i + 1 < len(train_loader):
                continue
            if scaler:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            epoch_float = epoch + (i + 1) / len(train_loader)
            if step % args.log_every == 0:
                now = time.perf_counter()
                row = {"step": step, "epoch": round(epoch_float, 3), "lr": scheduler.get_last_lr()[0],
                       **{f"loss_{k}": round(running[k] / max(running["n"], 1), 4) for k in ("total", "switch", "duration")},
                       "sec_per_step": round((now - t_log) / args.log_every, 3)}
                append_row(out_dir / "train_log.csv", row)
                log(f"step {step:>6}/{total_steps}  epoch {epoch_float:.2f}  loss {row['loss_total']:.4f}  "
                    f"switch {row['loss_switch']:.4f}  duration {row['loss_duration']:.4f}  lr {row['lr']:.2e}  {row['sec_per_step']:.2f}s/step")
                running = {"total": 0.0, "switch": 0.0, "duration": 0.0, "n": 0}
                t_log = now
            if args.eval_every and step % args.eval_every == 0:
                dev_eval(step, epoch_float)
                t_log = time.perf_counter()
            if args.max_steps and step >= args.max_steps:
                done = True
                break
            if args.time_budget_min and (time.perf_counter() - t_start) / 60 > args.time_budget_min:
                log(f"time budget of {args.time_budget_min:g} min reached at step {step} (epoch {epoch_float:.2f}); stopping training")
                stopped_by_budget = done = True
                break
        if not (args.eval_every and step % args.eval_every == 0):
            dev_eval(step, float(epoch + 1) if not done else epoch_float)
            t_log = time.perf_counter()
        if done:
            break
    train_minutes = (time.perf_counter() - t_start) / 60
    peak_gb = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(n_gpus)] if device.type == "cuda" else []
    log(f"training finished: {step} steps in {train_minutes:.1f} min ({60 * train_minutes / max(step, 1):.2f} s/step), "
        f"{nonfinite} batches skipped for a non-finite loss, peak GPU memory {peak_gb} GB")

    # ------------------------------------------------------------------ final evaluation with the best checkpoint
    if best["state"] is not None and best["step"] != step:
        model.load_state_dict(best["state"])
        log(f"restored the best checkpoint (step {best['step']}, dev {args.select_by} {best['score']:.4f})")
    elif args.save_model and not (out_dir / "model.pt").exists():
        save_model(model, out_dir / "model.pt", {"step": step})
    dev = evaluate_convs(runner, dev_full, tok, device, amp_dtype, loss_fn, batch_size=args.eval_batch_size,
                         max_len=args.max_len, stride=args.stride)
    test = evaluate_convs(runner, test_convs, tok, device, amp_dtype, loss_fn, thresholds=dev["thresholds"],
                          batch_size=args.eval_batch_size, max_len=args.max_len, stride=args.stride)
    for name, res in (("dev", dev), ("test", test)):
        with open(out_dir / f"{name}_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0]))
            w.writeheader()
            w.writerows(res["rows"])

    cols = ["pair", "switch_rate_%", "threshold", "P", "R", "F1", "F1_at_0.5", "dur_acc", "F1_macro_by_type"]
    table = markdown_table(cols, [[r.get(c, "") for c in cols] for r in test["rows"]])
    s = test["summary"]
    text = (f"# {args.run_name or args.model}: test split\n\n{table}\n\n"
            f"Mean Anticipatory F1 = {s['mean_F1']:.4f}, sigma = {s['sigma_F1']:.4f} "
            f"(min {s['min_F1']:.3f} {s['min_pair']}, max {s['max_F1']:.3f} {s['max_pair']}); "
            f"mean F1 at threshold 0.5 = {s['mean_F1_at_0.5']:.4f}; duration accuracy at true switches = {s['mean_dur_acc']:.4f}; "
            f"macro F1 by switch type = {s['mean_F1_macro_by_type']:.4f}.\n")
    (out_dir / "results.md").write_text(text, encoding="utf-8")
    log("\n" + text)

    summary = {
        "run_name": args.run_name or args.model, "model": args.model, "backbone": model.backbone_name, "out_dir": str(out_dir),
        "steps": step, "epochs": args.epochs, "train_minutes": round(train_minutes, 1), "best_step": best["step"],
        "nonfinite_batches": nonfinite, "stopped_by_time_budget": stopped_by_budget, "peak_gpu_memory_gb": peak_gb, "max_len": args.max_len, "stride": args.stride,
        "lambda_switch": args.lambda_switch, "lambda_duration": args.lambda_duration,
        "switch_weight": args.switch_weight, "duration_weight": args.duration_weight, "train_top_layers": args.train_top_layers,
        "max_train_convs": args.max_train_convs, "dev": dev["summary"], "dev_losses": dev["losses"],
        "test": test["summary"], "test_losses": test["losses"], "thresholds": dev["thresholds"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {out_dir}/summary.json")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
