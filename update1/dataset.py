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

Subword alignment: each word is tokenized on its own with the model's tokenizer, so no
subword can span two words (a merged token would let position t peek at word t+1). The
word keeps the spacing it had in the original text (`space_before` from split.py): Latin,
Arabic, Korean ... words get their leading space, while Mandarin, Cantonese and Japanese
words are glued together as in the original, so the tokenizer sees natural text. Every
subword inherits the language of its word (`lang_ids`). The prediction labels y_sw / y_dur
sit on the LAST subword of each word, because that is the first position where the whole
word has been seen; all other subwords get -1 (ignored).

Both views start with the same BOS token (`<s>` for XLM-R; Qwen2.5 has none), so the model
sees identical inputs during training and streaming inference.

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

    return AutoTokenizer.from_pretrained(TOKENIZERS.get(name_or_key, name_or_key))


def bos_id(tok) -> int | None:
    """Token put in front of every window and every streaming prefix (None for Qwen2.5)."""
    return tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id


class WordEncoder:
    """Tokenizes one word at a time, with or without the space in front of it.

    SentencePiece tokenizers (XLM-R) mark a preceding space with "▁" and add it to every
    string they encode, so a word that follows without a space is encoded by the subword
    model directly, bypassing that pre-tokenizer. Byte-level BPE tokenizers (Qwen2.5)
    encode the space as part of the next token, so the space is simply written in front.
    """

    def __init__(self, tok):
        self.tok = tok
        first = tok.convert_ids_to_tokens(tok("hello", add_special_tokens=False)["input_ids"])[0]
        self.metaspace = first.startswith("▁")
        self.cache: dict[tuple[str, bool], list[int]] = {}

    def __call__(self, word: str, space: bool) -> list[int]:
        key = (word, space)
        if key not in self.cache:
            if self.metaspace and not space:
                backend = self.tok.backend_tokenizer
                text = backend.normalizer.normalize_str(word) if backend.normalizer else word
                ids = [t.id for t in backend.model.tokenize(text)]
            else:
                text = " " + word if (space and not self.metaspace) else word
                ids = self.tok(text, add_special_tokens=False)["input_ids"]
            self.cache[key] = ids
        return self.cache[key]


_ENCODERS: dict[int, WordEncoder] = {}


def word_encoder(tok) -> WordEncoder:
    if id(tok) not in _ENCODERS:
        _ENCODERS[id(tok)] = WordEncoder(tok)
    return _ENCODERS[id(tok)]


def encode_words(words: list[str], space_before: list[bool] | None, tok) -> tuple[list[int], list[int]]:
    """Return (input_ids, word_ids) with every word tokenized separately.

    The first word is encoded as the start of a text: with "▁" for XLM-R, without a space
    for Qwen2.5, which is what both tokenizers do with a whole sentence.
    """
    enc = word_encoder(tok)
    if space_before is None:  # data written before split.py stored the spacing
        space_before = [True] * len(words)
    ids, wids = [], []
    for w, (word, sp) in enumerate(zip(words, space_before)):
        piece = enc(word, sp if w > 0 else enc.metaspace)
        if not piece:  # a word that normalizes away entirely still needs a position for its label
            piece = [tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id]
        ids.extend(piece)
        wids.extend([w] * len(piece))
    return ids, wids


def lang_id(lang: str, pair: str) -> int:
    """0 = matrix language (L1) of the pair, 1 = embedded language (L2)."""
    return 0 if lang == PAIRS[pair][0] else 1


def encode_conversation(conv: dict, tok) -> dict:
    """Tokenize word by word and align languages and labels to subwords."""
    ids, wids = encode_words(conv["words"], conv.get("space_before"), tok)
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
        bos = bos_id(tok)
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
    """Streaming view: after each word t yield (t, prefix_ids, y_sw[t], y_dur[t]).

    The prefix starts with the same BOS token as the training windows."""
    e = encode_conversation(conv, tok)
    bos = [bos_id(tok)] if bos_id(tok) is not None else []
    for w, last in sorted(e["last_sub_of_word"].items()):
        yield w, bos + e["input_ids"][: last + 1], conv["y_sw"][w], conv["y_dur"][w]


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
