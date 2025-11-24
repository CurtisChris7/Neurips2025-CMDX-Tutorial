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

class ShawStyleMHA(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, max_relative_position=16, qkv_bias=False, auto_regression=False):
        super().__init__()
        assert (d_out % num_heads == 0)

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length

        self.clip = max_relative_position

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

        # Relative position embeddings
        self.key_relative_positions_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.head_dim)
        )
        self.value_relative_positions_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.head_dim)
        )

    def _generate_relative_positions_matrix(self, length):
        range_vec = torch.arange(length)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        if self.clip:
            distance_mat = torch.clamp(distance_mat, -self.clip, self.clip)
        return distance_mat 

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        assert num_tokens <= self.context_length

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

        # Compute the first part of Shaw encoding
        relative_matrix = self._generate_relative_positions_matrix(num_tokens)
        a_k = self.key_relative_positions_embeddings[relative_matrix + self.clip]
        a_v = self.value_relative_positions_embeddings[relative_matrix + self.clip]

        attn_scores = (queries @ keys.transpose(2, 3)) + torch.einsum('bhid,ijd->bhij', queries, a_k)

        if self.auto_regression:
            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values) + torch.einsum('bhij,ijd->bhid', attn_weights, a_v)
        context_vec.transpose_(1, 2) 

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection
        return context_vec

