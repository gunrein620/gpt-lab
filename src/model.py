# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)
        ))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, drop_rate: float = 0.1, qkv_bias: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias)
        self.ffn = FeedForward(d_model, drop_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding = InputEmbedding(
            config["vocab_size"], config["emb_dim"],
            config["context_length"], config["drop_rate"]
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(config["emb_dim"], config["n_heads"],
                           config["drop_rate"], config["qkv_bias"])
            for _ in range(config["n_layers"])
        ])
        self.norm = LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        return loss, logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx