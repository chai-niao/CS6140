"""Evaluate a saved run and check that streaming inference matches the training view.

    python evaluate.py --run_dir runs/xlmr_full                    # test metrics with the dev thresholds of the run
    python evaluate.py --run_dir runs/xlmr_full --streaming_check 5

Streaming check: the model is fed one conversation word by word (update1.dataset.iter_prefixes),
exactly like a keyboard receiving words, and P(switch) after each word is compared with the
probability that the one-pass training window gives at the same position. Because attention
is causal the two must agree, which shows that the predictions use the prefix only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

import setup_path  # noqa: F401
from common import markdown_table
from data import available_pairs, load_convs
from dataset import PrefixWindowDataset, bos_id, encode_conversation, get_tokenizer, iter_prefixes
from engine import amp_dtype_for, evaluate_convs, get_device
from model import load_model


@torch.no_grad()
def streaming_consistency(model, convs: list[dict], tok, device, n: int = 5, max_subwords: int = 200) -> list[dict]:
    """For up to n short conversations: max |P_stream(switch) - P_window(switch)| over all words."""
    model.eval()
    out = []
    for conv in convs:
        e = encode_conversation(conv, tok)
        if len(e["input_ids"]) > max_subwords:
            continue
        window = PrefixWindowDataset([conv], tok, max_len=max_subwords + 8, stride=max_subwords)[0]
        ids = torch.tensor([window["input_ids"]], device=device)
        sw_logits, _ = model(ids, torch.ones_like(ids))
        p_window = torch.softmax(sw_logits.float(), -1)[0, :, 1]
        worst = 0.0
        words = 0
        for _, prefix, _, _ in iter_prefixes(conv, tok):
            ids_p = torch.tensor([prefix], device=device)
            sw_p, _ = model(ids_p, torch.ones_like(ids_p))
            p_stream = torch.softmax(sw_p.float(), -1)[0, -1, 1]
            worst = max(worst, abs(float(p_stream) - float(p_window[len(prefix) - 1])))
            words += 1
        out.append({"conv_id": conv["conv_id"], "words": words, "subwords": len(e["input_ids"]), "max_abs_diff": worst})
        if len(out) >= n:
            break
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
    ap.add_argument("--streaming_check", type=int, default=0, help="number of conversations for the streaming consistency check")
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
        checks = streaming_consistency(model, convs, tok, device, n=args.streaming_check)
        lines = [f"Streaming consistency on {len(checks)} conversations ({config['model']}, fp32, {device}):",
                 markdown_table(["conv_id", "words", "subwords", "max |P_stream - P_window|"],
                                [[c["conv_id"], c["words"], c["subwords"], f"{c['max_abs_diff']:.2e}"] for c in checks])]
        worst = max((c["max_abs_diff"] for c in checks), default=0.0)
        lines.append(f"\nworst difference {worst:.2e}: {'identical up to floating point noise' if worst < 1e-3 else 'MISMATCH'}")
        text = "\n".join(lines)
        print(text)
        (run_dir / "streaming_check.md").write_text(text + "\n", encoding="utf-8")
        result["streaming_max_abs_diff"] = worst
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
