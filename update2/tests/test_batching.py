"""Length-bucketed batch sampler: every window once per epoch, similar lengths per batch, new order per epoch."""
import random

from batching import LengthBucketBatchSampler


def test_bucket_sampler_covers_every_window_once():
    rng = random.Random(1)
    lengths = [rng.randint(5, 257) for _ in range(1003)]
    sampler = LengthBucketBatchSampler(lengths, batch_size=32, bucket_batches=10, seed=0)
    epoch1 = list(sampler)
    assert len(epoch1) == len(sampler)
    assert sorted(i for b in epoch1 for i in b) == list(range(len(lengths)))
    assert all(1 <= len(b) <= 32 for b in epoch1)
    epoch2 = list(sampler)
    assert epoch1 != epoch2 and sorted(i for b in epoch2 for i in b) == list(range(len(lengths)))


def test_bucketing_reduces_padding():
    rng = random.Random(2)
    lengths = [rng.randint(5, 257) for _ in range(3200)]
    padded = lambda batches: sum(max(lengths[i] for i in b) * len(b) for b in batches)  # noqa: E731
    bucketed = padded(list(LengthBucketBatchSampler(lengths, 32, bucket_batches=50, seed=0)))
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    plain = padded([idx[i:i + 32] for i in range(0, len(idx), 32)])
    assert bucketed < 0.7 * plain
