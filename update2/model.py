"""Dual-head model: a causal transformer backbone with a switch head and a duration head.

    h_t              = Backbone(x_1 .. x_t)              (causal: position t sees only the prefix)
    z_t              = Dropout(LayerNorm(h_t))
    switch logits    = W_sw z_t   in R^2                 (y_sw: 0 = no switch, 1 = switch at t+1)
    duration logits  = W_dur z_t  in R^3                 (y_dur: burst bin 0 / 1 / 2, -1 = ignored)

Backbones:
    Model A  "xlmr"  xlm-roberta-base, an encoder made causal with config.is_decoder = True
    Model B  "qwen"  Qwen/Qwen2.5-0.5B, a decoder that is causal by construction
Any other Hugging Face model name is accepted; encoder types get the causal mask.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

BACKBONES = {"xlmr": "xlm-roberta-base", "qwen": "Qwen/Qwen2.5-0.5B"}
# bidirectional by default: the causal attention mask has to be switched on
ENCODER_TYPES = {"xlm-roberta", "roberta", "bert"}
N_DURATION_BINS = 3


def tiny_config(key: str, vocab_size: int):
    """A two-layer random backbone of the same family, for unit tests and smoke runs."""
    if key == "xlmr":
        from transformers import XLMRobertaConfig

        return XLMRobertaConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                intermediate_size=64, max_position_embeddings=1030, type_vocab_size=1, pad_token_id=1)
    if key == "qwen":
        from transformers import Qwen2Config

        return Qwen2Config(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           num_key_value_heads=2, intermediate_size=64, max_position_embeddings=1030)
    raise ValueError(f"no tiny config for {key!r}")


class DualHeadModel(nn.Module):
    def __init__(self, backbone: str = "xlmr", dropout: float = 0.1, tiny_vocab: int | None = None):
        super().__init__()
        name = BACKBONES.get(backbone, backbone)
        self.backbone_key = backbone
        self.backbone_name = name
        config = tiny_config(backbone, tiny_vocab) if tiny_vocab is not None else AutoConfig.from_pretrained(name)
        self.is_encoder_type = config.model_type in ENCODER_TYPES
        if self.is_encoder_type:
            config.is_decoder = True  # causal attention mask (Model A)
        config.use_cache = False
        if tiny_vocab is not None:
            self.backbone = AutoModel.from_config(config)
        elif self.is_encoder_type:
            self.backbone = AutoModel.from_pretrained(name, config=config, add_pooling_layer=False)
        else:
            self.backbone = AutoModel.from_pretrained(name, config=config)
        self.backbone.float()  # train in fp32 master weights whatever the checkpoint dtype; AMP is applied outside
        hidden = config.hidden_size
        # a LayerNorm in front of the heads puts both backbones on the same scale (Qwen's final hidden
        # states are much larger than XLM-R's), so the randomly initialised heads start from a sane loss
        self.head_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.switch_head = nn.Linear(hidden, 2)
        self.duration_head = nn.Linear(hidden, N_DURATION_BINS)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(self.head_norm(h))
        return self.switch_head(h), self.duration_head(h)

    # ---------------------------------------------------------------- parameter groups
    def transformer_layers(self) -> list[nn.Module]:
        b = self.backbone
        if hasattr(b, "encoder") and hasattr(b.encoder, "layer"):  # XLM-R / BERT family
            return list(b.encoder.layer)
        if hasattr(b, "layers"):  # Qwen2 / Llama family
            return list(b.layers)
        raise AttributeError(f"cannot find the transformer layers of {type(b).__name__}")

    def train_top_layers_only(self, k: int) -> None:
        """Freeze the backbone except its last k transformer layers (and the final norm); heads stay trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers = self.transformer_layers()
        for layer in layers[max(len(layers) - k, 0):]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "norm"):  # Qwen2 final RMSNorm
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def count_parameters(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def describe(self) -> str:
        total, trainable = self.count_parameters()
        how = "causal mask via is_decoder=True" if self.is_encoder_type else "causal by construction"
        return (f"{self.backbone_name} ({type(self.backbone).__name__}, {how}), "
                f"{len(self.transformer_layers())} layers, hidden {self.switch_head.in_features}, "
                f"{total / 1e6:.1f}M parameters, {trainable / 1e6:.1f}M trainable")


def save_model(model: DualHeadModel, path, extra: dict | None = None) -> None:
    """Checkpoint in fp16 to halve the file size; the heads and all weights are restored to fp32 by load_model."""
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "backbone": model.backbone_key, **(extra or {})}, path)


def load_model(path, backbone: str | None = None, tiny_vocab: int | None = None, map_location="cpu") -> DualHeadModel:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = DualHeadModel(backbone or ckpt["backbone"], tiny_vocab=tiny_vocab)
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["state_dict"].items()})
    return model
