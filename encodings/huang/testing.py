import torch

from HuangAttention import HuangAttention
from HuangM1 import HuangM1
from HuangM2 import HuangM2
from HuangM3 import HuangM3


# -----------------------------
# Common config
# -----------------------------
BATCH_SIZE = 1
SEQ_LEN = 6
D_IN = 8
D_OUT = 8
NUM_HEADS = 2
CONTEXT_LEN = 16
MAX_REL_POS = 3


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# 1. Forward pass tests
# -----------------------------

def test_huang_forward_pass_method4():
    print("\nRunning test_huang_forward_pass_method4 (Method 4 / HuangAttention)...")
    torch.manual_seed(42)

    model = HuangAttention(
        d_in=D_IN,
        d_out=D_OUT,
        context_length=CONTEXT_LEN,
        dropout=0.0,
        num_heads=NUM_HEADS,
        max_relative_position=MAX_REL_POS,
        qkv_bias=True,
        auto_regression=False,
    ).to(device)

    model.eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_IN).to(device)
    output = model(x)

    print("Output shape:", output.shape)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_OUT), "Output shape mismatch!"
    print("✅ Forward pass (Method 4) passed.")


def test_huang_forward_pass_method1():
    print("\nRunning test_huang_forward_pass_method1 (Method 1 / HuangM1)...")
    torch.manual_seed(42)

    model = HuangM1(
        d_in=D_IN,
        d_out=D_OUT,
        context_length=CONTEXT_LEN,
        dropout=0.0,
        num_heads=NUM_HEADS,
        max_relative_position=MAX_REL_POS,
        qkv_bias=True,
        auto_regression=False,
    ).to(device)

    model.eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_IN).to(device)
    output = model(x)

    print("Output shape:", output.shape)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_OUT), "Output shape mismatch!"
    print("✅ Forward pass (Method 1) passed.")


def test_huang_forward_pass_method2():
    print("\nRunning test_huang_forward_pass_method2 (Method 2 / HuangM2)...")
    torch.manual_seed(42)

    model = HuangM2(
        d_in=D_IN,
        d_out=D_OUT,
        context_length=CONTEXT_LEN,
        dropout=0.0,
        num_heads=NUM_HEADS,
        max_relative_position=MAX_REL_POS,
        qkv_bias=True,
        auto_regression=False,
    ).to(device)

    model.eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_IN).to(device)
    output = model(x)

    print("Output shape:", output.shape)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_OUT), "Output shape mismatch!"
    print("✅ Forward pass (Method 2) passed.")


def test_huang_forward_pass_method3():
    print("\nRunning test_huang_forward_pass_method3 (Method 3 / HuangM3)...")
    torch.manual_seed(42)

    model = HuangM3(
        d_in=D_IN,
        d_out=D_OUT,
        context_length=CONTEXT_LEN,
        dropout=0.0,
        num_heads=NUM_HEADS,
        max_relative_position=MAX_REL_POS,
        qkv_bias=True,
        auto_regression=False,
    ).to(device)

    model.eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_IN).to(device)
    output = model(x)

    print("Output shape:", output.shape)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_OUT), "Output shape mismatch!"
    print("✅ Forward pass (Method 3) passed.")


# -----------------------------
# 2. Relative embedding EFFECT tests
# -----------------------------

def _relative_effect_test(model_class, rel_attr_name, method_name):
    print(f"\nRunning relative effect test for {method_name} ...")
    torch.manual_seed(0)

    model = model_class(
        d_in=D_IN,
        d_out=D_OUT,
        context_length=CONTEXT_LEN,
        dropout=0.0,
        num_heads=NUM_HEADS,
        max_relative_position=MAX_REL_POS,
        qkv_bias=True,
        auto_regression=False,
    )

    model.eval()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_IN)

    with torch.no_grad():
        output_orig = model(x).detach()

        rel_param = getattr(model, rel_attr_name)
        rel_param += 10.0  # large perturbation

        output_modified = model(x).detach()

    diff = (output_orig - output_modified).abs().mean().item()
    print(f"Mean |Δoutput| after modifying {rel_attr_name}:", diff)
    assert diff > 1e-4, f"Relative embeddings in {method_name} did not affect output!"
    print(f"✅ Relative embedding effect test passed for {method_name}.")


def test_relative_effect_method4():
    # HuangAttention: attribute is `relative_parameters`
    _relative_effect_test(
        HuangAttention,
        "relative_parameters",
        "Method 4 / HuangAttention"
    )


def test_relative_effect_method1():
    # HuangM1: attribute is `key_relative_positions_embeddings`
    _relative_effect_test(
        HuangM1,
        "key_relative_positions_embeddings",
        "Method 1 / HuangM1"
    )


def test_relative_effect_method2():
    _relative_effect_test(
        HuangM2,
        "key_relative_positions_embeddings",
        "Method 2 / HuangM2"
    )


def test_relative_effect_method3():
    _relative_effect_test(
        HuangM3,
        "key_relative_positions_embeddings",
        "Method 3 / HuangM3"
    )


# -----------------------------
# 3. Relative embedding GRADIENT tests
# -----------------------------

def _relative_grad_test(model_class, rel_attr_name, method_name):
    print(f"\nRunning relative gradient test for {method_name} ...")
    torch.manual_seed(0)

    model = model_class(
        d_in=D_IN,
        d_out=D_OUT,
        context_length=CONTEXT_LEN,
        dropout=0.0,
        num_heads=NUM_HEADS,
        max_relative_position=MAX_REL_POS,
        qkv_bias=True,
        auto_regression=False,
    )

    model.train()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_IN)

    output = model(x)
    loss = output.sum()
    loss.backward()

    rel_param = getattr(model, rel_attr_name)
    grad = rel_param.grad

    grad_norm = grad.norm().item() if grad is not None else 0.0
    print(f"Gradient norm for {rel_attr_name} in {method_name}: {grad_norm}")

    assert grad is not None and grad.abs().sum() > 0, \
        f"No gradient for {rel_attr_name} in {method_name}!"
    print(f"✅ Relative embedding gradient flow test passed for {method_name}.")


def test_relative_grad_method4():
    _relative_grad_test(
        HuangAttention,
        "relative_parameters",
        "Method 4 / HuangAttention"
    )


def test_relative_grad_method1():
    _relative_grad_test(
        HuangM1,
        "key_relative_positions_embeddings",
        "Method 1 / HuangM1"
    )


def test_relative_grad_method2():
    _relative_grad_test(
        HuangM2,
        "key_relative_positions_embeddings",
        "Method 2 / HuangM2"
    )


def test_relative_grad_method3():
    _relative_grad_test(
        HuangM3,
        "key_relative_positions_embeddings",
        "Method 3 / HuangM3"
    )


# -----------------------------
# 4. Manual ground-truth test for Method 4 (Eq. 16)
# -----------------------------

def test_huang_method4_manual_ground_truth():
    """
    Sets HuangAttention (Method 4) to a simple, interpretable configuration
    and verifies that the model output matches a dynamically computed
    manual implementation of Eq. (16).

    This is analogous to set_manual_params_and_verify for ShawStyleMHA.
    """
    print("\nRunning test_huang_method4_manual_ground_truth...")
    torch.manual_seed(0)

    # Small, easy-to-reason-about setup
    d_in = d_out = 4
    num_heads = 2
    head_dim = d_out // num_heads
    max_rel_pos = 1
    seq_len = 3
    batch_size = 1

    model = HuangAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=seq_len,
        dropout=0.0,
        num_heads=num_heads,
        max_relative_position=max_rel_pos,
        qkv_bias=True,
        auto_regression=False,   # no mask for simplicity
    )

    # --- Manual example input ---
    x = torch.tensor([[
        [1.0, 1.1, 1.2, 1.3],
        [1.5, 1.6, 1.7, 1.8],
        [2.0, 2.1, 2.2, 2.3],
    ]])  # (1, 3, 4)

    # --- Set model parameters to simple forms (identity-like) ---
    with torch.no_grad():
        I = torch.eye(d_out)

        model.W_query.weight.copy_(I)
        model.W_query.bias.zero_()

        model.W_key.weight.copy_(I)
        model.W_key.bias.zero_()

        model.W_value.weight.copy_(I)
        model.W_value.bias.zero_()

        model.out_proj.weight.copy_(I)
        model.out_proj.bias.zero_()

        # Relative vectors: all ones * 0.05
        rel_vec = torch.ones(2 * max_rel_pos + 1, head_dim) * 0.05
        model.relative_parameters.copy_(rel_vec)

    model.eval()

    # --- Compute manual reference (Eq. 16) ---
    with torch.no_grad():
        b, T, _ = x.shape

        Q = model.W_query(x).view(b, T, num_heads, head_dim).transpose(1, 2)  # (b,h,T,d)
        K = model.W_key(x).view(b, T, num_heads, head_dim).transpose(1, 2)
        V = model.W_value(x).view(b, T, num_heads, head_dim).transpose(1, 2)

        rel_mat = model._generate_relative_positions_matrix(T)  # (T,T), in [-1,1]
        a = model.relative_parameters[rel_mat + model.clip]      # (T,T,d)

        QK = Q @ K.transpose(2, 3)                               # (b,h,T,T)
        Qa = torch.einsum("bhid,ijd->bhij", Q, a)                # (b,h,T,T)
        Ka = torch.einsum("bhjd,ijd->bhij", K, a)                # (b,h,T,T)
        scores = QK + Qa + Ka

        attn_weights = torch.softmax(scores / (head_dim ** 0.5), dim=-1)
        context = attn_weights @ V                               # (b,h,T,d)
        context = context.transpose(1, 2).contiguous().view(b, T, d_out)
        manual_output = model.out_proj(context)

    # --- Compare with model forward ---
    with torch.no_grad():
        output = model(x)

    max_diff = (output - manual_output).abs().max().item()
    print("Manual ground truth output:\n", manual_output)
    print("Model output:\n", output)
    print(f"Max difference between model output and manual ground truth: {max_diff:.6e}")

    assert max_diff < 3e-5, "❌ Mismatch between HuangAttention and manual Eq.16 implementation!"
    print("✅ HuangAttention matches manual Eq.16 ground truth.")


# -----------------------------
# Main entry point
# -----------------------------
if __name__ == "__main__":
    # 1. Forward-pass tests
    test_huang_forward_pass_method4()
    test_huang_forward_pass_method1()
    test_huang_forward_pass_method2()
    test_huang_forward_pass_method3()

    # 2. Relative effect tests
    test_relative_effect_method4()
    test_relative_effect_method1()
    test_relative_effect_method2()
    test_relative_effect_method3()

    # 3. Gradient tests
    test_relative_grad_method4()
    test_relative_grad_method1()
    test_relative_grad_method2()
    test_relative_grad_method3()

    # 4. Manual ground truth (Method 4)
    test_huang_method4_manual_ground_truth()

    print("\nAll Huang tests completed successfully.\n")
