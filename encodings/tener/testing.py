import torch
from TenerAttention import TenerAttention

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def test_forward_shape():
    torch.manual_seed(0)
    d_in = 8
    d_out = 8
    context_len = 16
    num_heads = 2

    model = TenerAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_len,
        dropout=0.0,
        num_heads=num_heads,
    ).to(device)
    model.eval()

    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_in).to(device)
    out = model(x)

    assert out.shape == (batch_size, seq_len, d_out), \
        f"Expected shape {(batch_size, seq_len, d_out)}, got {out.shape}"
    print("test_forward_shape: OK")


def test_no_nan_inf():
    torch.manual_seed(1)
    d_in = 8
    d_out = 8
    context_len = 16
    num_heads = 2

    model = TenerAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_len,
        dropout=0.0,
        num_heads=num_heads,
    )
    model.eval()

    x = torch.randn(2, 6, d_in)
    out = model(x)

    assert not torch.isnan(out).any().item(), "Output contains NaNs"
    assert not torch.isinf(out).any().item(), "Output contains Infs"
    print("test_no_nan_inf: OK")


def test_eval_deterministic():
    torch.manual_seed(2)
    d_in = 8
    d_out = 8
    context_len = 16
    num_heads = 2

    model = TenerAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_len,
        dropout=0.0,
        num_heads=num_heads,
    )
    model.eval()

    x = torch.randn(3, 12, d_in)
    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1, out2, atol=1e-6), \
        "Eval mode with dropout=0 should be deterministic"
    print("test_eval_deterministic: OK")


def test_equation_faithfulness():
    """
    Re-implement the math inside TenerAttention.forward and check that
    it matches the module output (up to numerical error).
    This matches your current implementation exactly.
    """
    torch.manual_seed(3)
    d_in = 8
    d_out = 8
    context_len = 16
    num_heads = 2

    model = TenerAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_len,
        dropout=0.0,
        num_heads=num_heads,
    )
    model.eval()

    B, T = 2, 6
    x = torch.randn(B, T, d_in)

    with torch.no_grad():
        out_module = model(x)

        # Rebuild Q, K, V
        Q = model.W_query(x)  # (B, T, d_out)
        K = model.W_key(x)
        V = model.W_value(x)

        H = model.num_heads
        D = model.head_dim

        # (B, T, d_out) -> (B, H, T, D)
        Q = Q.view(B, T, H, D).transpose(1, 2)
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)

        # Relative matrix
        R = model.R[:T, :T, :]  # (T, T, D)

        # Terms
        QK = Q @ K.transpose(2, 3)                           # (B, H, T, T)
        QR = torch.einsum("bhid,ijd->bhij", Q, R)            # (B, H, T, T)
        uK = torch.einsum("hd,bhjd->bhj", model.u, K).unsqueeze(2)  # (B, H, 1, T)
        vR = torch.einsum("hd,ijd->hij", model.v, R).unsqueeze(0)   # (1, H, T, T)

        attn_scores = QK + QR + uK + vR                      # (B, H, T, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)    # no scaling in your impl

        context = attn_weights @ V                           # (B, H, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, H * D)
        out_manual = model.out_proj(context)                 # (B, T, d_out)

    assert torch.allclose(out_module, out_manual, atol=1e-6), \
        "Manual equation reimplementation does not match module output"
    print("test_equation_faithfulness: OK")


def main():
    test_forward_shape()
    test_no_nan_inf()
    test_eval_deterministic()
    test_equation_faithfulness()
    print("All TenerAttention tests passed.")


if __name__ == "__main__":
    main()
