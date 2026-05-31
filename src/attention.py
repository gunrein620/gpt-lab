# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model # emb_dim = d_model = 각 토큰을 표현하는 벡터 크기
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        self.W_qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=qkv_bias)
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(drop_rate)
        

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        # [1] Q, K, V 만들기
        qkv = self.W_qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        # [2] head 분리
        B, T, C = x.shape
        
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # [3] attention score 계산
        attn_score = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)

        # [4] causal mask 사용
        if causal_mask:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_score = attn_score.masked_fill(mask, float('-inf'))

        # [5] softmax -> attention weight
        attn_weight = F.softmax(attn_score, dim=-1)

        # [6] attention weight @ V
        attn_output = attn_weight @ V

        # [7] head 합치기
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # [8] output projection + dropout
        out = self.output_projection(attn_output)
        out = self.dropout(out)

        if return_attention_weights:
            return out, attn_weight
        return out
