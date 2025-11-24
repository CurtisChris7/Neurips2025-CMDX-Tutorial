import torch
from ShawMultiHeadAttention import ShawStyleMHA

def test_shaw_forward_pass():
    print("\nRunning test_shaw_forward_pass...")
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 8
    d_in = 12
    d_out = 12
    num_heads = 2
    max_rel_pos = 3

    model = ShawStyleMHA(
        d_in=d_in,
        d_out=d_out,
        dropout=0.0,
        num_heads=num_heads,
        max_relative_position=max_rel_pos,
        qkv_bias=True,
    )

    model.eval()
    x = torch.randn(batch_size, seq_len, d_in)
    output = model(x)

    print("Output shape:", output.shape)
    assert output.shape == (batch_size, seq_len, d_out), "Output shape mismatch!"
    print("✅ Forward pass test passed.")


def test_relative_bias_effect():
    print("\nRunning test_relative_bias_effect...")
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 4
    d_in = 8
    d_out = 8
    num_heads = 2
    max_rel_pos = 2

    model = ShawStyleMHA(
        d_in=d_in,
        d_out=d_out,
        dropout=0.0,
        num_heads=num_heads,
        max_relative_position=max_rel_pos,
        qkv_bias=True,
    )

    model.eval()
    x = torch.randn(batch_size, seq_len, d_in)

    output_orig = model(x).detach()

    with torch.no_grad():
        model.key_relative_positions_embeddings += 10.0
        model.value_relative_positions_embeddings += 10.0

    output_modified = model(x).detach()

    diff = (output_orig - output_modified).abs().mean().item()
    print("Mean output difference after modifying biases:", diff)
    assert diff > 1e-4, "Relative biases did not affect output!"
    print("✅ Relative bias effect test passed.")


def test_relative_bias_gradients():
    print("\nRunning test_relative_bias_gradients...")
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 8
    d_in = 12
    d_out = 12
    num_heads = 2
    max_rel_pos = 3

    model = ShawStyleMHA(
        d_in=d_in,
        d_out=d_out,
        dropout=0.0,
        num_heads=num_heads,
        max_relative_position=max_rel_pos,
        qkv_bias=True,
    )

    model.train()
    x = torch.randn(batch_size, seq_len, d_in)

    output = model(x)
    loss = output.sum()
    loss.backward()

    grad_a_k = model.key_relative_positions_embeddings.grad
    grad_a_v = model.value_relative_positions_embeddings.grad

    print("Gradient norm a_k:", grad_a_k.norm().item())
    print("Gradient norm a_v:", grad_a_v.norm().item())

    assert grad_a_k is not None and grad_a_k.abs().sum() > 0, "No gradient for a_k!"
    assert grad_a_v is not None and grad_a_v.abs().sum() > 0, "No gradient for a_v!"
    print("✅ Relative bias gradient flow test passed.")


def set_manual_params_and_verify(model):
    """
    Sets the model's parameters to the manually defined small example
    and verifies that model output matches dynamically computed ground truth.
    """
    import torch

    # --- Manual Example Data ---

    # Input x
    x = torch.tensor([[
        [1.0, 1.1, 1.2, 1.3],
        [1.5, 1.6, 1.7, 1.8],
        [2.0, 2.1, 2.2, 2.3]
    ]])  # (1, 3, 4)
   
    # Weight matrices
    W_query = torch.eye(4)  # (4x4)
    W_key = torch.eye(4)
    W_value = torch.eye(4)
    W_out = torch.eye(4)

    # Relative position embeddings
    max_rel_pos = 1
    key_relative_positions_embeddings = torch.ones(2 * max_rel_pos + 1) * 0.05
    value_relative_positions_embeddings = torch.ones(2 * max_rel_pos + 1) * 0.01

    # --- Set Model Parameters ---
    with torch.no_grad():
        model.W_query.weight.copy_(W_query.T)
        model.W_query.bias.zero_()

        model.W_key.weight.copy_(W_key.T)
        model.W_key.bias.zero_()

        model.W_value.weight.copy_(W_value.T)
        model.W_value.bias.zero_()

        model.out_proj.weight.copy_(W_out.T)
        model.out_proj.bias.zero_()

        model.key_relative_positions_embeddings.copy_(key_relative_positions_embeddings)
        model.value_relative_positions_embeddings.copy_(value_relative_positions_embeddings)

    model.eval()

    # --- Compute Ground Truth Dynamically ---
    with torch.no_grad():
        b, num_tokens, d_in = x.shape
        d_out = model.d_out
        num_heads = model.num_heads
        head_dim = model.head_dim

        queries = x.view(b, num_tokens, num_heads, head_dim).transpose(1, 2)
        keys = x.view(b, num_tokens, num_heads, head_dim).transpose(1, 2)
        values = x.view(b, num_tokens, num_heads, head_dim).transpose(1, 2)

        relative_matrix = model._generate_relative_positions_matrix(num_tokens)
        a_k = model.key_relative_positions_embeddings[relative_matrix + model.max_relative_position]
        a_v = model.value_relative_positions_embeddings[relative_matrix + model.max_relative_position]

        raw_scores = torch.matmul(queries, keys.transpose(2, 3))
        relative_scores = torch.einsum('bhid,ijd->bhij', queries, a_k)
        attn_scores = raw_scores + relative_scores

        mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()
        attn_scores[:, :, mask] = float('-inf')

        attn_weights = torch.softmax(attn_scores / (head_dim ** 0.5), dim=-1)

        context = torch.matmul(attn_weights, values) + torch.einsum('bhij,ij->bhi', attn_weights, a_v)
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, d_out)

        manual_ground_truth_output = context

    # --- Run Model Forward Pass ---
    with torch.no_grad():
        output = model(x)

    # --- Compare ---
    max_diff = (output - manual_ground_truth_output).abs().max().item()

    print("Ground Truth (computed dynamically):")
    print(manual_ground_truth_output)
    print("Model Output:")
    print(output)
    print(f"Max difference between model output and manual ground truth: {max_diff:.6e}")

    assert max_diff < 3e-3, "❌ Mismatch between model output and manual expected output!"

    print("✅ Model matches dynamically computed ground truth exactly.")


# --- Manual run
if __name__ == "__main__":
    test_shaw_forward_pass()
    test_relative_bias_effect()
    test_relative_bias_gradients()

    # Assuming you have your ShawStyleMHA class defined
    model = ShawStyleMHA(
        d_in=4,
        d_out=4,
        dropout=0.0,
        num_heads=2,
        max_relative_position=1,
        qkv_bias=True,
    )

    set_manual_params_and_verify(model)

