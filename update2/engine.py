"""Shared pieces of train.py and evaluate.py: device and mixed precision, batched prediction,
and the per-split evaluation that turns predictions into the metrics of metrics.py."""
from __future__ import annotations

import contextlib

import numpy as np
import torch

import setup_path  # noqa: F401
from data import pair_of, switch_type_short
from dataset import make_loader
from metrics import pair_rows, summarize, tune_thresholds


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype_for(device: torch.device, amp="auto"):
    """Mixed precision dtype: 'auto' | 'bf16' | 'fp16' | '0'. 'auto' picks bf16 on GPUs with native bf16
    (compute capability 8.0 or newer) and fp16 on older GPUs such as the T4, where torch reports bf16 as
    supported but runs it through slow emulation. No mixed precision on CPU / MPS."""
    amp = str(amp).lower()
    if amp in ("0", "none", "false", "off") or device.type != "cuda":
        return None
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast(device: torch.device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device: torch.device, amp_dtype=None, cs_type_of: dict | None = None, loss_fn=None):
    """Run the model over a loader.

    Returns (records, losses): records has one entry per labelled position (y_sw != -1) with
    pair, cs_type, p_switch = P(switch), y_sw, dur_pred = argmax of the duration head, y_dur;
    losses are the mean switch / duration / total loss over batches (empty dict without loss_fn)."""
    model.eval()
    cs_type_of = cs_type_of or {}
    pairs, types, probs, y_sws, dur_preds, y_durs = [], [], [], [], [], []
    sums = {"total": 0.0, "switch": 0.0, "duration": 0.0}
    n_batches = 0
    for batch in loader:
        b = to_device(batch, device)
        with autocast(device, amp_dtype):
            sw_logits, dur_logits = model(b["input_ids"], b["attention_mask"])
        if loss_fn is not None:
            l = loss_fn(sw_logits, dur_logits, b["y_sw"], b["y_dur"])
            for k in sums:
                sums[k] += float(l[k])
            n_batches += 1
        mask = b["y_sw"] >= 0
        p = torch.softmax(sw_logits.float(), dim=-1)[..., 1][mask]
        d = dur_logits.argmax(dim=-1)[mask]
        rows = mask.nonzero(as_tuple=False)[:, 0].tolist()  # window index of every labelled position, row-major like the mask
        conv_ids = batch["conv_id"]
        pairs.extend(pair_of(conv_ids[r]) for r in rows)
        types.extend(cs_type_of.get(conv_ids[r], "unknown") for r in rows)
        probs.append(p.cpu().numpy().astype(np.float32))
        y_sws.append(b["y_sw"][mask].cpu().numpy().astype(np.int8))
        dur_preds.append(d.cpu().numpy().astype(np.int8))
        y_durs.append(b["y_dur"][mask].cpu().numpy().astype(np.int8))
    records = {
        "pair": np.array(pairs, dtype=object),
        "cs_type": np.array(types, dtype=object),
        "p_switch": np.concatenate(probs) if probs else np.zeros(0, np.float32),
        "y_sw": np.concatenate(y_sws) if y_sws else np.zeros(0, np.int8),
        "dur_pred": np.concatenate(dur_preds) if dur_preds else np.zeros(0, np.int8),
        "y_dur": np.concatenate(y_durs) if y_durs else np.zeros(0, np.int8),
    }
    losses = {k: v / n_batches for k, v in sums.items()} if n_batches else {}
    return records, losses


def evaluate_convs(model, convs: list[dict], tok, device: torch.device, amp_dtype=None, loss_fn=None,
                   thresholds: dict[str, float] | None = None, batch_size: int = 32, max_len: int = 128, stride: int = 64,
                   loader=None):
    """Metrics of `model` on a list of conversations.

    Without `thresholds` the per-pair decision thresholds are tuned on these conversations
    (dev split); with them, they are applied as they are (test split). `loader` reuses windows built
    once for `convs` (the periodic dev evaluation) instead of tokenizing them again.
    Returns dict(rows, summary, thresholds, losses, records)."""
    if loader is None:
        loader = make_loader(convs, tok, batch_size=batch_size, max_len=max_len, stride=stride, shuffle=False)
    cs_type_of = {c["conv_id"]: c.get("cs_type", "unknown") for c in convs}
    records, losses = predict(model, loader, device, amp_dtype, cs_type_of, loss_fn)
    if thresholds is None:
        thresholds = tune_thresholds(records)
    rows = pair_rows(records, thresholds, switch_type_short)
    return {"rows": rows, "summary": summarize(rows), "thresholds": thresholds, "losses": losses, "records": records}


class AutocastForward(torch.nn.Module):
    """Runs the wrapped module under torch.autocast. Needed with nn.DataParallel, whose replicas run
    in worker threads where the autocast state of the main thread does not apply."""

    def __init__(self, module: torch.nn.Module, device_type: str, dtype):
        super().__init__()
        self.module = module
        self.device_type = device_type
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        if self.dtype is None:
            return self.module(*args, **kwargs)
        with torch.autocast(device_type=self.device_type, dtype=self.dtype):
            return self.module(*args, **kwargs)
