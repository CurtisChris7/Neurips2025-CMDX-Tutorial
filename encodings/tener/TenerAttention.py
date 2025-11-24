"""
Disclaimer: This is an implementation for educational reference.

This module provides an implementation designed for conceptual clarity and
faithful exposition of the underlying mathematical ideas. Its primary purpose
is to illustrate the mechanics of the original method in a transparent and
developer-friendly manner. The design choices intentionally prioritize
readability and explicitness so that the core concepts remain approachable to
students, researchers, and developers exploring the technique—rather than
computational efficiency or engineering optimizations.

While correct in behavior and faithful to the conceptual formulation, this
implementation does not aim to match the efficiency of industrial or
production-grade systems. Official or large-scale implementations may rely on
compressed representations, fused operations, or implicit algorithmic shortcuts 
that are intentionally omitted here in the interest of clarity.

Author: Christopher Curtis <curtis.ch@northeastern.edu>
"""
import torch
import torch.nn as nn

class TenerAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, base=10000.0, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.base = base
        self.u = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
        self.v = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
        self.register_buffer("R", self.create_rotation_matrix(context_length))

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

    def get_rotation_vectors(self, angles):
        d = self.head_dim
        pos_idx = torch.arange(0, d, 2, dtype=angles.dtype)
        div_term = self.base ** (-pos_idx.float() / d)
        angle_rates = angles.unsqueeze(-1) * torch.repeat_interleave(div_term, 2)

        # Now build R
        R = torch.zeros(*angle_rates.shape)
        R[..., 0::2] = torch.sin(angle_rates[..., 0::2])
        R[..., 1::2] = torch.cos(angle_rates[..., 1::2])
        return R

    def create_rotation_matrix(self, context_length):
        # Assign the vector to every cell in the matrix
        pos = torch.arange(context_length)
        rel_pos = pos.view(1, -1) - pos.view(-1, 1)
        rotation_matrix = self.get_rotation_vectors(rel_pos)
        return rotation_matrix
    

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        R = self.R[:num_tokens,:num_tokens,:]
        attn_scores= (queries @ keys.transpose(2,3)).add_(
                torch.einsum('bhid,ijd->bhij', queries, R)
            ).add_(
                torch.einsum('hd,bhjd->bhj', self.u, keys).unsqueeze(2)  # (B,H,N) -> (B,H,1,N)
            ).add_(torch.einsum('hd,ijd->hij', self.v, R))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection
        return context_vec
