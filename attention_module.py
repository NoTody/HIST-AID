# Reference LLaMA for RoPE implementation: https://github.com/facebookresearch/llama/blob/main/llama/model.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def precompute_freqs_cis(dim: int, t: torch.Tensor, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        t (torch.Tensor): Normalized time offset tensor of shape [B, L].
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    B = t.shape[0]
    device = t.device
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)).repeat(B, 1).to(device)
    freqs = torch.einsum('bf,bh->bfh', (t, freqs))
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SelfAttention(nn.Module):
    def __init__(self, num_embed, num_heads=12, dropout=0., bias=False, fused_attn=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_embed = num_embed
        self.fused_attn = fused_attn

        # qkv projection
        self.qkv = nn.Linear(num_embed, num_embed * 3, bias=bias)

        head_dim = num_embed // num_heads
        self.scale = head_dim ** -0.5

        # dropout
        self.dropout = dropout

    def forward(self, x, attn_mask, freqs_cis):
        B, N, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        
        # split heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if freqs_cis != None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        
        # attention (scaled dot product)
        attn = None
        if self.fused_attn:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, \
                    dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            y = attn @ v
            
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, N, C)
            
        return y, attn

# MLP
class MLPBlock(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate=0.):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

# Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, num_embed, num_heads=12, dropout=0., bias=False):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_embed = num_embed

        # qkv projection
        self.Wq = nn.Linear(num_embed, num_embed, bias=bias)
        self.Wkv = nn.Linear(num_embed, num_embed * 2, bias=bias)

        # dropout
        self.dropout = dropout

    def calc_cross_attn(self, x1, x2, attn_mask):
        B1, N1, C1 = x1.shape # batch size, sequence length, embedding dimensionality (n_embd)
        B2, N2, C2 = x2.shape # batch size, sequence length, embedding dimensionality (n_embd)
        
        # split heads
        q = self.Wq(x1).reshape(B1, N1, 1, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        
        kv = self.Wkv(x2).reshape(B2, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # attention (scaled dot product)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, \
            dropout_p=self.dropout if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B1, N1, C1) # re-assemble all head outputs side by side

        return y

    def forward(self, x1, x2, attn_mask):
        y = self.calc_cross_attn(x1, x2, attn_mask)
        return y

# Multi-Head Cross Attention
class CrossAttentionBlock(nn.Module):
    def __init__(self, num_embed, num_heads=12, dropout=0., bias=False):
        super().__init__()
        
        # multi-head self-attention
        self.msa = SelfAttention(num_embed, num_heads, dropout, bias)
        
        # multi-head cross-attention
        self.mca = CrossAttention(num_embed, num_heads, dropout, bias)

        # multi-layer perceptron projection
        self.mlp1 = MLPBlock(num_embed, num_embed*4, dropout_rate=dropout)
        self.mlp2 = MLPBlock(num_embed, num_embed*4, dropout_rate=dropout)
        
        # layer norm
        self.norm_mca1, self.norm_mca2 = nn.LayerNorm(num_embed, eps=1e-6), \
            nn.LayerNorm(num_embed, eps=1e-6)

        self.norm_mlp1, self.norm_mlp2 = nn.LayerNorm(num_embed, eps=1e-6), \
            nn.LayerNorm(num_embed, eps=1e-6)
            
        self.norm_msa2 = nn.LayerNorm(num_embed, eps=1e-6)

    def forward(self, x, attn_mask_ca, attn_mask_sa):
        x1, x2 = x[0], x[1]
        
        # cross attention
        mca_out = self.mca(self.norm_mca1(x1), self.norm_mca2(x2), attn_mask_ca)
        x1 = x1 + mca_out
        x1 = x1 + self.mlp1(self.norm_mlp1(x1))
        
        # self attention
        x2 = x2 + self.msa(self.norm_msa2(x2), attn_mask_sa, freqs_cis=None)
        x2 = x2 + self.mlp2(self.norm_mlp2(x2))
        
        return x1, x2

# Cross Attention Layer
class CrossAttentionLayer(nn.Module):
    def __init__(self, num_embed, n_layers=3, num_heads=12, dropout=0., bias=False):
        super().__init__()
        
        self.n_layers = n_layers
        self.mca_blocks = CrossAttentionBlock(num_embed, num_heads, dropout, bias)
        
    def forward(self, x, attn_mask_ca, attn_mask_sa):
        x1, x2 = x[0], x[1]

        # multihead cross attention
        x1, x2 = self.mca_blocks([x1, x2], attn_mask_ca, attn_mask_sa)
        out = x1[:, 0, :]
        
        return out