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
import math

# Taken directly from ALiBi implementation (press et al.)
def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
    else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


class AlibiAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Causal Mask
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
            
        # Precompute slopes (ALiBi paper choice)
        self.register_buffer("alibi_slopes", torch.tensor(get_slopes(num_heads)))


    def forward(self, x):
        b, num_tokens, _ = x.size()

        q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = q @ k.transpose(-2, -1)

        # ALiBi Bias = slopes * position difference
        pos = torch.arange(num_tokens, device=x.device)
        # H -> H,1,1 and N,N -> 1,N,N
        alibi = self.alibi_slopes.view(self.num_heads,1,1) * (pos.view(1, -1) - pos.view(-1, 1)).abs().unsqueeze(0)
        
        attn_scores -= alibi  # Note the minus (distance penalty)

        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], float('-inf'))

        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)
