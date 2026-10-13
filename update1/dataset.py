"""Step 4: the Streaming DataLoader (PyTorch) with subword alignment.

Two views of the same labelled conversations:

  PrefixWindowDataset  training view. The subword stream of a conversation is cut into
                       windows of `max_len` subwords with stride `stride`; every window
                       carries labels for its positions, and a causal model trained on a
                       window predicts, at each position, the switch at the next word.
                       Because attention is causal, position j only sees x_1..x_j of the
                       window, so one forward pass scores every prefix at once.

  iter_prefixes        inference / demo view. Yields the growing prefix one word at a
                       time, exactly like a keyboard receiving words.

Subword alignment: each word is tokenized with the model's own tokenizer. Every subword
inherits the language of its word (`lang_ids`). The prediction labels y_sw / y_dur sit on
the LAST subword of each word, because that is the first position where the whole word
has been seen; all other subwords get -1 (ignored).

Run `python dataset.py --demo` to print an alignment table, batch shapes and a streaming
trace with both tokenizers (XLM-R and Qwen2.5). Output also goes to results/dataloader_demo.txt.
"""
from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout

import torch
from torch.utils.data import DataLoader, Dataset

from common import RESULTS_DIR, load_split
from lid import PAIRS

IGNORE = -1
TOKENIZERS = {"xlmr": "xlm-roberta-base", "qwen": "Qwen/Qwen2.5-0.5B"}


def get_tokenizer(name_or_key: str):
    from transformers import AutoTokenizer

    name = TOKENIZERS.get(name_or_key, name_or_key)
    kwargs = {"add_prefix_space": True} if "qwen" in name.lower() else {}
    return AutoTokenizer.from_pretrained(name, **kwargs)


def lang_id(lang: str, pair: str) -> int:
    """0 = matrix language (L1) of the pair, 1 = embedded language (L2)."""
    return 0 if lang == PAIRS[pair][0] else 1


def encode_conversation(conv: dict, tok) -> dict:
    """Tokenize word by word and align languages and labels to subwords."""
    enc = tok(conv["words"], is_split_into_words=True, add_special_tokens=False)
    ids = enc["input_ids"]
    wids = enc.word_ids()
    n = len(ids)
    sub_lang = [lang_id(conv["langs"][w], conv["pair"]) for w in wids]
    y_sw = [IGNORE] * n
    y_dur = [IGNORE] * n
    last_sub_of_word = {}
    for i, w in enumerate(wids):
        last_sub_of_word[w] = i  # overwritten until the last subword of w
    for w, i in last_sub_of_word.items():
        y_sw[i] = conv["y_sw"][w]
        y_dur[i] = conv["y_dur"][w]
    return {"input_ids": ids, "word_ids": wids, "lang_ids": sub_lang, "y_sw": y_sw, "y_dur": y_dur,
            "last_sub_of_word": last_sub_of_word}


class PrefixWindowDataset(Dataset):
    def __init__(self, convs: list[dict], tok, max_len: int = 128, stride: int = 64):
        self.items: list[dict] = []
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
        for conv in convs:
            e = encode_conversation(conv, tok)
            n = len(e["input_ids"])
            start = 0
            first = True
            while True:
                end = min(start + max_len, n)
                sl = slice(start, end)
                item = {k: list(e[k][sl]) for k in ("input_ids", "lang_ids", "y_sw", "y_dur")}
                if not first:  # positions already labelled by the previous window are ignored here
                    keep_from = max_len - stride
                    for k in ("y_sw", "y_dur"):
                        item[k] = [IGNORE] * min(keep_from, len(item[k])) + item[k][keep_from:]
                if bos is not None:
                    item["input_ids"] = [bos] + item["input_ids"]
                    item["lang_ids"] = [0] + item["lang_ids"]
                    item["y_sw"] = [IGNORE] + item["y_sw"]
                    item["y_dur"] = [IGNORE] + item["y_dur"]
                item["conv_id"] = conv["conv_id"]
                self.items.append(item)
                if end >= n:
                    break
                start += stride
                first = False

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        L = max(len(b["input_ids"]) for b in batch)

        def pad(key, value):
            return torch.tensor([b[key] + [value] * (L - len(b[key])) for b in batch])

        return {
            "input_ids": pad("input_ids", pad_id),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in batch]),
            "lang_ids": pad("lang_ids", 0),
            "y_sw": pad("y_sw", IGNORE),
            "y_dur": pad("y_dur", IGNORE),
            "conv_id": [b["conv_id"] for b in batch],
        }

    return collate


def make_loader(convs: list[dict], tok, batch_size: int = 16, max_len: int = 128, stride: int = 64, shuffle: bool = True) -> DataLoader:
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ds = PrefixWindowDataset(convs, tok, max_len=max_len, stride=stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(pad_id))


def iter_prefixes(conv: dict, tok):
    """Streaming view: after each word t yield (t, prefix_ids, y_sw[t], y_dur[t])."""
    e = encode_conversation(conv, tok)
    for w, last in sorted(e["last_sub_of_word"].items()):
        yield w, e["input_ids"][: last + 1], conv["y_sw"][w], conv["y_dur"][w]


# ------------------------------------------------------------------ demo
def alignment_table(conv: dict, tok, max_words: int = 12) -> str:
    e = encode_conversation(conv, tok)
    rows = ["| t | word | lang | subwords | y_sw | y_dur |", "|---|---|---|---|---|---|"]
    for w in range(min(max_words, len(conv["words"]))):
        subs = [tok.convert_ids_to_tokens(e["input_ids"][i]) for i, wid in enumerate(e["word_ids"]) if wid == w]
        rows.append(f"| {w + 1} | {conv['words'][w]} | {conv['langs'][w]} | {' '.join(subs)} | {conv['y_sw'][w]} | {conv['y_dur'][w]} |")
    return "\n".join(rows)


def demo(pair: str, split: str, keys: list[str]) -> None:
    convs = load_split(pair, split)
    conv = next(c for c in convs if sum(1 for v in c["y_sw"] if v == 1) >= 1)
    print(f"pair = {pair}, split = {split}, conversations = {len(convs)}, demo conversation = {conv['conv_id']} ({len(conv['words'])} words)\n")
    for key in keys:
        tok = get_tokenizer(key)
        print(f"=== tokenizer: {TOKENIZERS[key]}  (vocab {len(tok):,}) ===")
        print(alignment_table(conv, tok))
        e = encode_conversation(conv, tok)
        print(f"\nwords {len(conv['words'])}, subwords {len(e['input_ids'])}, "
              f"labelled positions {sum(1 for v in e['y_sw'] if v >= 0)} (one per word), "
              f"subwords per word {len(e['input_ids']) / len(conv['words']):.2f}")
        loader = make_loader(convs[:64], tok, batch_size=8, max_len=128, stride=64)
        batch = next(iter(loader))
        print("\nfirst batch:")
        for k in ("input_ids", "attention_mask", "lang_ids", "y_sw", "y_dur"):
            print(f"  {k:15s} {tuple(batch[k].shape)}  dtype={batch[k].dtype}")
        n_lab = int((batch["y_sw"] >= 0).sum())
        n_sw = int((batch["y_sw"] == 1).sum())
        print(f"  labelled positions in batch: {n_lab}, of which switches: {n_sw}; windows in dataset: {len(loader.dataset)}")
        print("\nstreaming trace (first 8 words):")
        for t, prefix, y_sw, y_dur in iter_prefixes(conv, tok):
            if t >= 8:
                break
            print(f"  after word {t + 1:2d} ({conv['words'][t]!r:>16}): prefix = {len(prefix):3d} subwords -> targets y_sw={y_sw:2d} y_dur={y_dur:2d}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--pair", default="Spanish_eng")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tokenizers", nargs="*", default=["xlmr", "qwen"])
    args = ap.parse_args()
    if not args.demo:
        ap.print_help()
        return
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo(args.pair, args.split, args.tokenizers)
    text = buf.getvalue()
    sys.stdout.write(text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "dataloader_demo.txt").write_text(text, encoding="utf-8")
    print(f"wrote {RESULTS_DIR / 'dataloader_demo.txt'}")


if __name__ == "__main__":
    main()
