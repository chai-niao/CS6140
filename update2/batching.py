"""Length-bucketed training batches.

Most SwitchLingua conversations are much shorter than a 256-subword window, so with plain random batches
every batch is padded to its longest window and about half of the compute goes into padding. The sampler
below shuffles the windows, cuts them into chunks of `bucket_batches` batches, sorts every chunk by length
and yields the batches in random order (group-by-length batching, as in fairseq and the Hugging Face
Trainer). Every epoch gets a new order.
"""
from __future__ import annotations

import random

from torch.utils.data import DataLoader, Sampler

import setup_path  # noqa: F401
from dataset import PrefixWindowDataset, make_collate


class LengthBucketBatchSampler(Sampler):
    def __init__(self, lengths: list[int], batch_size: int, bucket_batches: int = 50, seed: int = 0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.chunk = batch_size * max(bucket_batches, 1)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        n, out = len(self.lengths), 0
        for c in range(0, n, self.chunk):
            m = min(self.chunk, n - c)
            out += (m + self.batch_size - 1) // self.batch_size
        return out

    def __iter__(self):
        rng = random.Random(self.seed * 1000 + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for c in range(0, len(idx), self.chunk):
            chunk = sorted(idx[c:c + self.chunk], key=self.lengths.__getitem__)
            batches += [chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size)]
        rng.shuffle(batches)
        yield from batches


def make_train_loader(convs: list[dict], tok, batch_size: int, max_len: int, stride: int,
                      bucket_batches: int = 50, seed: int = 0) -> DataLoader:
    """Training loader over prefix windows; bucket_batches = 0 gives plain random batches."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ds = PrefixWindowDataset(convs, tok, max_len=max_len, stride=stride)
    if bucket_batches <= 0:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=make_collate(pad_id))
    lengths = [len(item["input_ids"]) for item in ds.items]
    sampler = LengthBucketBatchSampler(lengths, batch_size, bucket_batches, seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate(pad_id))
