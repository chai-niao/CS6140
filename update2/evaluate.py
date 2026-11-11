"""Evaluate a saved run and check that streaming inference matches the training view.

    python evaluate.py --run_dir runs/xlmr_full                    # test metrics with the dev thresholds of the run
    python evaluate.py --run_dir runs/xlmr_full --streaming_check 2   # 2 random + the longest conversation per pair

Streaming check: the model is fed one conversation word by word (update1.dataset.iter_prefixes),
exactly like a keyboard receiving words, and P(switch) after each word is compared with the
probability that the one-pass training window gives at the same position. Because attention
is causal the two must agree, which shows that the predictions use the prefix only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

import setup_path  # noqa: F401
from common import markdown_table
from data import available_pairs, load_convs, pair_of
from dataset import PrefixWindowDataset, encode_conversation, get_tokenizer, iter_prefixes
from engine import amp_dtype_for, evaluate_convs, get_device
from model import load_model


def pick_streaming_convs(convs: list[dict], n_per_pair: int, seed: int = 0) -> list[dict]:
    """n random conversations of every pair plus the longest one of the pair, so that every tokenizer path
    (with and without spaces between words, romanized Hindi, ...) and conversations that span several
    training windows are checked."""
    by_pair: dict[str, list[dict]] = {}
    for c in convs:
        by_pair.setdefault(pair_of(c["conv_id"]), []).append(c)
    out = []
    for pair, cs in by_pair.items():
        rng = random.Random(f"{seed}-{pair}")
        chosen = rng.sample(cs, min(n_per_pair, len(cs)))
        longest = max(cs, key=lambda c: len(c["words"]))
        if longest not in chosen:
            chosen.append(longest)
        out.extend(chosen)
    return out


@torch.no_grad()
def streaming_consistency(model, convs: list[dict], tok, device, max_len: int, stride: int) -> list[dict]:
    """For every conversation: max |P_stream(switch) - P_window(switch)| over its labelled words.

    P_window comes from the training / evaluation windows (PrefixWindowDataset, max_len and stride of the run);
    P_stream feeds the words one at a time with iter_prefixes, whose context is capped at the window that scores
    the word. Both views use only the prefix, so they must agree for short and long conversations alike."""
    model.eval()
    out = []
    for conv in convs:
        e = encode_conversation(conv, tok)
        p_window = []
        for item in PrefixWindowDataset([conv], tok, max_len=max_len, stride=stride):  # windows in order
            ids = torch.tensor([item["input_ids"]], device=device)
            sw_logits, _ = model(ids, torch.ones_like(ids))
            labelled = torch.tensor(item["y_sw"], device=device) >= 0
            p_window.extend(torch.softmax(sw_logits.float(), -1)[0, :, 1][labelled].tolist())
        p_stream, longest_prefix = [], 0
        for _, prefix, y_sw, _ in iter_prefixes(conv, tok, max_len=max_len, stride=stride):
            if y_sw < 0:  # the last word has no target and is not scored by the windows either
                continue
            ids_p = torch.tensor([prefix], device=device)
            sw_p, _ = model(ids_p, torch.ones_like(ids_p))
            p_stream.append(float(torch.softmax(sw_p.float(), -1)[0, -1, 1]))
            longest_prefix = max(longest_prefix, len(prefix))
        if len(p_stream) != len(p_window):
            raise RuntimeError(f"{conv['conv_id']}: {len(p_stream)} streaming vs {len(p_window)} window predictions")
        worst = max((abs(a - b) for a, b in zip(p_stream, p_window)), default=0.0)
        out.append({"conv_id": conv["conv_id"], "words": len(conv["words"]), "subwords": len(e["input_ids"]),
                    "windows": max(1, -(-max(len(e["input_ids"]) - max_len, 0) // stride) + 1),
                    "longest_prefix": longest_prefix, "max_abs_diff": worst})
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="directory written by train.py (model.pt, config.json, summary.json)")
    ap.add_argument("--data_dir", default=None, help="default: the data_dir of the run")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--max_convs", type=int, default=0, help="conversations per pair (0 = all)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", help="auto | bf16 | fp16 | 0")
    ap.add_argument("--streaming_check", type=int, default=0,
                    help="random conversations per pair for the streaming consistency check (the longest of each pair is added)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_dir = args.data_dir or config["data_dir"]
    device = get_device(args.device)
    amp_dtype = amp_dtype_for(device, args.amp)
    tok = get_tokenizer(config["model"])
    model = load_model(run_dir / "model.pt", config["model"], tiny_vocab=len(tok) if config.get("tiny") else None).to(device)
    print(model.describe())

    pairs = args.pairs or available_pairs(data_dir)
    convs = load_convs(data_dir, args.split, pairs, args.max_convs)
    thresholds = None
    if args.split == "test" and (run_dir / "summary.json").exists():
        thresholds = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("thresholds")
    res = evaluate_convs(model, convs, tok, device, amp_dtype, thresholds=thresholds, batch_size=args.batch_size,
                         max_len=config.get("max_len", 128), stride=config.get("stride", 64))
    out = run_dir / f"{args.split}_metrics_eval.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rows"][0]))
        w.writeheader()
        w.writerows(res["rows"])
    cols = ["pair", "threshold", "P", "R", "F1", "F1_at_0.5", "dur_acc", "F1_macro_by_type"]
    print(markdown_table(cols, [[r.get(c, "") for c in cols] for r in res["rows"]]))
    print(json.dumps(res["summary"], indent=2))
    print(f"wrote {out}")

    result = {"summary": res["summary"]}
    if args.streaming_check:
        model.float()
        max_len, stride = config.get("max_len", 128), config.get("stride", 64)
        picked = pick_streaming_convs(convs, args.streaming_check)
        checks = streaming_consistency(model, picked, tok, device, max_len, stride)
        lines = [f"Streaming consistency ({config['model']}, fp32, {device}): {len(checks)} conversations, "
                 f"{args.streaming_check} random + the longest of each of {len({pair_of(c['conv_id']) for c in checks})} pairs; "
                 f"windows of {max_len} subwords, stride {stride}; the streaming prefix is capped at the window that scores the word.",
                 markdown_table(["conv_id", "words", "subwords", "windows", "longest prefix", "max |P_stream - P_window|"],
                                [[c["conv_id"], c["words"], c["subwords"], c["windows"], c["longest_prefix"], f"{c['max_abs_diff']:.2e}"]
                                 for c in checks])]
        worst = max((c["max_abs_diff"] for c in checks), default=0.0)
        multi = sum(1 for c in checks if c["windows"] > 1)
        lines.append(f"\n{multi} of {len(checks)} conversations span more than one window. "
                     f"Worst difference {worst:.2e}: {'identical up to floating point noise' if worst < 1e-3 else 'MISMATCH'}")
        text = "\n".join(lines)
        print(text)
        (run_dir / "streaming_check.md").write_text(text + "\n", encoding="utf-8")
        result["streaming_max_abs_diff"] = worst
        result["streaming_conversations"] = len(checks)
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
