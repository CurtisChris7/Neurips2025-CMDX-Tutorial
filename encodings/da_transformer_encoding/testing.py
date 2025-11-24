import math
import torch
from DAAttention import DAAttention

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def manual_forward(attn, x):
    """
    Manual DA-Transformer attention forward pass
    for equation-faithfulness checking.
    """
    b, t, d_in = x.shape
    H = attn.num_heads
    d_head = attn.head_dim
    d_out = attn.d_out

    # Q, K, V projections → reshape to (b, H, t, d_head)
    Q = attn.W_query(x).view(b, t, H, d_head).transpose(1, 2)
    K = attn.W_key(x).view(b, t, H, d_head).transpose(1, 2)
    V = attn.W_value(x).view(b, t, H, d_head).transpose(1, 2)

    # Distance matrix
    R = attn.get_relative_distance_matrix(t).to(x.device).float()   # (t, t)
    R = R.unsqueeze(0)                                              # (1, t, t)

    w = attn.w.view(H, 1, 1)
    v = attn.v.view(H, 1, 1)

    # Apply distance mapping
    R_weighted = w * R
    R_hat = (1 + torch.exp(v)) / (1 + torch.exp(v - R_weighted))   # (H, t, t)
                                   # (1, H, t, t)

    # Raw attention scores
    scores = torch.matmul(Q, K.transpose(2, 3))

    # ReLU and distance scaling
    scores = torch.relu(scores) * R_hat

    if attn.auto_regression:
        # Apply causal mask
        mask = attn.mask.bool()[:t, :t].to(x.device)
        scores = scores.masked_fill(mask, -torch.inf)

    # Softmax
    attn_weights = torch.softmax(scores / K.shape[-1]**0.5, dim=-1)

    # Multiply by V
    context = torch.matmul(attn_weights, V)                         # (b, H, t, d_head)

    # Recombine heads
    context = context.transpose(1, 2).contiguous().view(b, t, d_out)

    # Output projection
    return attn.out_proj(context)


# ---------------------------------------------------------
#                 TEST FUNCTIONS (VANILLA)
# ---------------------------------------------------------

def test_forward_shape():
    print("Running test_forward_shape()...")
    torch.manual_seed(0)

    attn = DAAttention(8, 8, context_length=8, num_heads=2, dropout=0.1).to(device)
    x = torch.randn(3, 5, 8).to(device)

    out = attn(x)

    assert out.shape == (3, 5, 8), "Output shape mismatch."
    assert torch.isfinite(out).all(), "NaNs or Infs detected."

    print("✓ test_forward_shape passed.")


def test_forward_faithfulness(auto_regression):
    print("Running test_forward_faithfulness()...")
    torch.manual_seed(123)

    attn = DAAttention(8, 8, context_length=6, num_heads=2, dropout=0.0, auto_regression=auto_regression)
    x = torch.randn(1, 6, 8)

    out_1 = attn(x)
    out_2 = manual_forward(attn, x)

    if not torch.allclose(out_1, out_2, atol=1e-5, rtol=1e-5):
        print("\n❌ Faithfulness test FAILED.")
        print("Max diff:", (out_1 - out_2).abs().max())
        raise AssertionError("Forward implementation does not match manual equation.")

    print("✓ test_forward_faithfulness passed.")


def test_backward_gradients():
    print("Running test_backward_gradients()...")
    torch.manual_seed(999)

    attn = DAAttention(8, 8, context_length=6, num_heads=2, dropout=0.1)
    x = torch.randn(2, 4, 8, requires_grad=True)

    out = attn(x).sum()
    out.backward()

    assert x.grad is not None, "No gradient for input."
    assert torch.isfinite(x.grad).all(), "Input gradient contains non-finite values."

    for name, param in attn.named_parameters():
        if param.grad is None:
            raise AssertionError(f"No gradient for parameter {name}")

        if not torch.isfinite(param.grad).all():
            raise AssertionError(f"Non-finite gradient in {name}")

    print("✓ test_backward_gradients passed.")


# ---------------------------------------------------------
#                   MAIN TEST RUNNER
# ---------------------------------------------------------

def main():
    print("=== Running DAAttention Tests (Pure Python) ===\n")

    test_forward_shape()
    test_forward_faithfulness(True)
    test_forward_faithfulness(False)
    test_backward_gradients()

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    main()
