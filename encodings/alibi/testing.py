import math
import torch

from AlibiAttention import AlibiAttention


def test_output_shape():
    torch.manual_seed(0)
    d_in = 16
    d_out = 16
    num_heads = 4
    context_length = 8
    batch_size = 2
    seq_len = 5

    model = AlibiAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=True,
    )

    x = torch.randn(batch_size, seq_len, d_in)
    out = model(x)

    assert out.shape == (batch_size, seq_len, d_out), (
        f"Unexpected output shape: {out.shape}, "
        f"expected {(batch_size, seq_len, d_out)}"
    )


def test_determinism_no_dropout():
    torch.manual_seed(1)
    d_in = 8
    d_out = 8
    num_heads = 2
    context_length = 6
    batch_size = 1
    seq_len = 4

    model = AlibiAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=False,
    )
    model.eval()

    x = torch.randn(batch_size, seq_len, d_in)

    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1, out2, atol=1e-6), \
        "Model is not deterministic in eval mode with dropout=0.0"


def test_slopes_properties():
    """
    Check that the registered slopes tensor:
    - has correct length
    - is positive
    - is strictly decreasing
    - spans a meaningful range
    """
    d_in = 8
    d_out = 8
    num_heads = 8
    context_length = 4

    model = AlibiAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=False,
    )

    slopes = model.alibi_slopes  # shape: (H,)
    assert slopes.ndim == 1, f"Expected 1D slopes, got shape {slopes.shape}"
    assert slopes.shape[0] == num_heads, "Slopes length != num_heads"
    assert torch.all(slopes > 0), "All slopes must be positive"

    # slopes should be strictly decreasing (head 0 has largest slope)
    diffs = slopes[:-1] - slopes[1:]
    assert torch.all(diffs > 0), (
        f"ALiBi slopes are not strictly decreasing: {slopes}"
    )

    ratio = (slopes[0] / slopes[-1]).item()
    assert ratio > 5.0, f"Expected a noticeable spread in slopes, got ratio={ratio:.4f}"


def test_causal_mask_no_future_leak():
    """
    Changing a future token should NOT change outputs at earlier positions.
    """
    torch.manual_seed(2)
    d_in = 8
    d_out = 8
    num_heads = 2
    context_length = 6
    batch_size = 1
    seq_len = 4

    model = AlibiAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=True,
    )
    model.eval()

    x_base = torch.randn(batch_size, seq_len, d_in)
    x_changed = x_base.clone()

    # Modify ONLY the last token
    x_changed[:, -1, :] += torch.randn_like(x_changed[:, -1, :])

    out_base = model(x_base)
    out_changed = model(x_changed)

    # Earlier positions must be identical (within numerical tolerance)
    assert torch.allclose(
        out_base[:, :-1, :],
        out_changed[:, :-1, :],
        atol=1e-6,
    ), (
        "Causal masking violated: earlier positions changed "
        "when only a future token changed."
    )


def test_sequence_len_smaller_than_context():
    """
    Ensure mask slicing and broadcasting work when seq_len < context_length.
    """
    torch.manual_seed(3)
    d_in = 8
    d_out = 8
    num_heads = 2
    context_length = 10
    batch_size = 1
    seq_len = 5

    model = AlibiAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=False,
    )

    x = torch.randn(batch_size, seq_len, d_in)
    out = model(x)

    assert out.shape == (batch_size, seq_len, d_out), \
        "Model failed when seq_len < context_length"


def test_gradients_flow_to_parameters():
    """
    Check that gradients flow to input and all key parameters.
    """
    torch.manual_seed(4)
    d_in = 8
    d_out = 8
    num_heads = 2
    context_length = 6
    batch_size = 2
    seq_len = 4

    model = AlibiAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=True,
    )

    x = torch.randn(batch_size, seq_len, d_in, requires_grad=True)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None, "No gradient flowed to the input."
    assert model.W_query.weight.grad is not None, "No grad for W_query.weight"
    assert model.W_key.weight.grad is not None, "No grad for W_key.weight"
    assert model.W_value.weight.grad is not None, "No grad for W_value.weight"
    assert model.out_proj.weight.grad is not None, "No grad for out_proj.weight"


def test_equation_faithfulness():
    """
    Manually reimplement the attention computation and compare to forward().
    This ensures:
    - mask is applied correctly
    - ALiBi bias is applied with the same broadcasting
    - scaling by sqrt(head_dim) matches
    """
    torch.manual_seed(5)
    d_in = 8
    d_out = 8
    num_heads = 2
    context_length = 6
    batch_size = 1
    seq_len = 4

    model = AlibiAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=True,
    )
    model.eval()

    x = torch.randn(batch_size, seq_len, d_in)
    out = model(x)

    # Manual reimplementation
    b, T, _ = x.shape
    H = num_heads
    head_dim = d_out // H

    Wq = model.W_query
    Wk = model.W_key
    Wv = model.W_value
    Wo = model.out_proj

    q = Wq(x).view(b, T, H, head_dim).transpose(1, 2)  # (b, H, T, d)
    k = Wk(x).view(b, T, H, head_dim).transpose(1, 2)
    v = Wv(x).view(b, T, H, head_dim).transpose(1, 2)

    scores = q @ k.transpose(-2, -1)  # (b, H, T, T)

    pos = torch.arange(T, device=x.device)
    diff = (pos.view(1, -1) - pos.view(-1, 1)).abs()  # (T, T)
    slopes = model.alibi_slopes.view(1, H, 1, 1)      # (1, H, 1, 1)
    alibi = slopes * diff.view(1, 1, T, T)            # (1, H, T, T)

    scores_manual = scores - alibi                    # (b, H, T, T)

    # apply same mask
    mask = model.mask[:T, :T]                         # (T, T)
    scores_manual = scores_manual.masked_fill(mask, float('-inf'))

    scores_manual = scores_manual / math.sqrt(head_dim)
    attn_manual = torch.softmax(scores_manual, dim=-1)
    # dropout=0.0, eval => no change

    context = attn_manual @ v                         # (b, H, T, d)
    context = context.transpose(1, 2).contiguous().view(b, T, d_out)
    out_manual = Wo(context)

    assert torch.allclose(out, out_manual, atol=1e-6), \
        "Manual attention computation does not match module.forward()"


def run_all_tests():
    tests = [
        test_output_shape,
        test_determinism_no_dropout,
        test_slopes_properties,
        test_causal_mask_no_future_leak,
        test_sequence_len_smaller_than_context,
        test_gradients_flow_to_parameters,
        test_equation_faithfulness,
    ]

    num_passed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"[PASS] {name}")
            num_passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            print(f"[ERROR] {name}: unexpected error: {e}")

    print(f"\n{num_passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all_tests()
