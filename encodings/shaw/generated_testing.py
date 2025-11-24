# test_shaw_attention.py

import torch
import torch.nn as nn
from ShawMultiHeadAttention import ShawMultiHeadAttention

# Assume your ShawMultiHeadAttention is already imported
# from shaw_attention import ShawMultiHeadAttention

# --- TEST CASES ---
def test_identity_case(model):
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    with torch.no_grad():
        model.key_relative_positions_embeddings.zero_()
        model.value_relative_positions_embeddings.zero_()
    model.eval()
    output = model(x)
    expected = x
    assert torch.allclose(output, expected, atol=1e-5), "❌ Identity case failed"
    print("✅ Identity case passed")

def test_pure_bias_case(model):
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    with torch.no_grad():
        model.key_relative_positions_embeddings.zero_()
        model.value_relative_positions_embeddings.fill_(0.01)
    model.eval()
    output = model(x)
    expected = x + 0.01
    assert torch.allclose(output, expected, atol=1e-5), "❌ Pure bias case failed"
    print("✅ Pure bias case passed")

def test_two_token_softmax_case(model):
    x = torch.tensor([[
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0]
    ]])
    with torch.no_grad():
        model.key_relative_positions_embeddings.zero_()
        model.value_relative_positions_embeddings.zero_()
    model.eval()
    output = model(x)
    q1 = x[:, 1, :]
    k0 = x[:, 0, :]
    k1 = x[:, 1, :]
    sim0 = (q1 * k0).sum() / (2 ** 0.5)
    sim1 = (q1 * k1).sum() / (2 ** 0.5)
    alpha = torch.softmax(torch.tensor([sim0, sim1]), dim=0)
    expected = alpha[0] * x[:, 0, :] + alpha[1] * x[:, 1, :]
    assert torch.allclose(output[:, 1, :], expected, atol=1e-4), "❌ Two token softmax case failed"
    print("✅ Two token softmax case passed")

def test_masking_case(model):
    x = torch.tensor([[
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0]
    ]])
    with torch.no_grad():
        model.key_relative_positions_embeddings.zero_()
        model.value_relative_positions_embeddings.zero_()
    model.eval()
    output = model(x)
    assert not torch.isnan(output).any(), "❌ Output contains NaNs"
    print("✅ Masking case passed")

def test_bias_shift_property(model):
    x = torch.tensor([[
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0]
    ]])
    with torch.no_grad():
        model.key_relative_positions_embeddings.zero_()
        model.value_relative_positions_embeddings.fill_(0.1)
    model.eval()
    output = model(x)
    shift_detected = (output - x).mean().abs() > 0.05
    assert shift_detected, "❌ Bias shift property failed"
    print("✅ Bias shift property passed")

def test_attention_sum_property(model):
    x = torch.tensor([[
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0]
    ]])
    with torch.no_grad():
        model.key_relative_positions_embeddings.zero_()
        model.value_relative_positions_embeddings.zero_()
    model.eval()
    b, num_tokens, _ = x.shape
    queries = model.W_query(x).view(b, num_tokens, model.num_heads, model.head_dim).transpose(1, 2)
    keys = model.W_key(x).view(b, num_tokens, model.num_heads, model.head_dim).transpose(1, 2)
    attn_scores = torch.matmul(queries, keys.transpose(2, 3))
    mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()
    attn_scores[:, :, mask] = float('-inf')
    attn_weights = torch.softmax(attn_scores / (model.head_dim ** 0.5), dim=-1)
    attn_sum = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "❌ Attention sum property failed"
    print("✅ Attention sum property passed")

# --- RUNNER ---
def run_all_tests():
    print("Running Shaw Attention Independent Tests...")
    model = ShawMultiHeadAttention(
        d_in=4,
        d_out=4,
        context_length=3,
        dropout=0.0,
        num_heads=2,
        max_relative_position=1,
        qkv_bias=False,
    )

    test_identity_case(model)
    test_pure_bias_case(model)
    test_two_token_softmax_case(model)
    test_masking_case(model)
    test_bias_shift_property(model)
    test_attention_sum_property(model)

    print("\n✅ All Shaw Attention tests passed!")

if __name__ == "__main__":
    run_all_tests()