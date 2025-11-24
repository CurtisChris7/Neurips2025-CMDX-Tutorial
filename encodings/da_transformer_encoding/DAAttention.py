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

class DAAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, auto_regression=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        self.auto_regression = auto_regression
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

        # DA Parameters
        self.w = nn.Parameter(torch.rand(self.num_heads))
        self.v = nn.Parameter(torch.rand(self.num_heads))


    def get_relative_distance_matrix(self, seq_len: int) -> torch.Tensor:
        # Row indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long)

        # Compute i - j for all pairs
        rel_dist = (positions[:, None] - positions[None, :]).abs()  # (seq_len, seq_len)
        return rel_dist

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # DA INTERVENTION HERE!
        R = self.get_relative_distance_matrix(num_tokens).to(x.device).unsqueeze(0) # (1,N,N)
        w = self.w.view(self.num_heads,1,1) # (H,1,1)
        v = self.v.view(self.num_heads,1,1) # (H,1,1)
        R_hat = (1 + torch.exp(v)) / (1 + torch.exp(v - (w*R)))

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = torch.relu(queries @ keys.transpose(2, 3)) * R_hat

        if self.auto_regression:
            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection
        return context_vec
