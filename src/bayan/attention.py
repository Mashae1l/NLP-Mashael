"""Lab 2: scaled dot-product attention and multi-head attention."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    """
    Scaled dot-product attention.

    Q × K^T
    ↓
    scale by sqrt(d_k)
    ↓
    apply mask
    ↓
    softmax
    ↓
    weights × V
    """

    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    weights = F.softmax(scores, dim=-1)

    return torch.matmul(weights, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        x = attention(q, k, v, mask)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)

        return self.out_proj(x)