import torch
from ricomodels.og_transformer.og_transformer_encoder import (
    OGPositionalEncoder,
    DotProductAttention,
    MultiHeadAttention,
)
from ricomodels.utils.predict_tools import allclose_replace_nan
import pytest


@pytest.fixture
def basic_config():
    return {
        "batch_size": 2,
        "sentence_length": 10,
        "embedding_size": 16,
        "max_sentence_length": 50,
        "num_queries": 3,
        "num_keys": 4,
        "qk_dim": 5,
        "v_dim": 6,
    }


def test_og_positional_encoder(basic_config):
    batch_size = basic_config["batch_size"]
    sentence_length = basic_config["sentence_length"]
    embedding_size = basic_config["embedding_size"]
    max_sentence_length = basic_config["max_sentence_length"]
    # Initialize the encoder
    encoder = OGPositionalEncoder(max_sentence_length, embedding_size)
    X = torch.rand(batch_size, sentence_length, embedding_size)
    output = encoder(X)

    # Check output shape
    assert output.shape == (
        batch_size,
        sentence_length,
        embedding_size,
    ), f"Output shape mismatch: {output.shape} != {(batch_size, sentence_length, embedding_size)}"
    print("Test passed!")

    computed_embedding = encoder.positional_embedding.squeeze(0)
    pos = torch.arange(0, max_sentence_length).unsqueeze(1)  # [max_sentence_length, 1]
    two_j = torch.arange(0, embedding_size).unsqueeze(
        0
    )  # [1, embedding_size] for broadcasting
    expected_embedding = torch.zeros_like(computed_embedding)
    angles = pos / (10000 ** (two_j // 2 * 2 / embedding_size))
    expected_embedding[:, 0::2] = torch.sin(angles[:, 0::2])  # Sine for even indices
    expected_embedding[:, 1::2] = torch.cos(angles[:, 1::2])  # Cosine for odd indices
    assert torch.allclose(
        computed_embedding, expected_embedding, atol=1e-6
    ), "Computed positional embedding does not match the expected values."


def test_scaled_dot_product_basic(basic_config):
    batch_size = basic_config["batch_size"]
    num_queries = basic_config["num_queries"]
    num_keys = basic_config["num_keys"]
    qk_dim = basic_config["qk_dim"]
    v_dim = basic_config["v_dim"]
    # Create random tensors
    q = torch.randn(batch_size, num_queries, qk_dim)
    k = torch.randn(batch_size, num_keys, qk_dim)
    v = torch.randn(batch_size, num_keys, v_dim)
    # No masking
    mask = torch.zeros(batch_size, num_queries, num_keys)
    attention = DotProductAttention()
    output = attention(q, k, v, mask)
    assert output.shape == (batch_size, num_queries, v_dim), "Output shape mismatch."


def test_scaled_dot_product_effectiveness():
    # Define queries and keys such that q[0] should attend more to k[0] and k[1], ignore k[2]
    q = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
    )  # Shape: [1, num_queries, 4]
    k = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
    )  # Shape: [1, num_keys, 4]
    v = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])  # Shape: [1, 3, 2]
    # Create mask: mask the third key for all queries
    mask = torch.tensor([[[0, 0, 1], [0, 0, 1]]])  # Shape: [1, num_queries, num_keys]

    attention = DotProductAttention()
    output = attention(q, k, v, attn_mask=mask)
    # Expected behavior:
    # For query 0: attends to key 0 and key 1
    # For query 1: attends to key 0 and key 1
    # Key 2 is masked and should have zero attention
    # Compute attention weights manually
    dk = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(dk, dtype=torch.float32)
    )
    masked_scores = scores + mask * -1e9
    attn_weights = torch.nn.functional.softmax(masked_scores, dim=-1)
    expected_output = torch.matmul(attn_weights, v)
    assert torch.allclose(
        output, expected_output, atol=1e-4
    ), "Masking effectiveness failed."


def test_scaled_dot_product_comparative_analysis():
    """
    This is the ultimate correctness check.
    """
    batch_size = 2
    num_queries = 4
    num_keys = 5
    qk_dim = 8
    v_dim = 8  # Set v_dim equal to qk_dim for comparison
    num_heads = 1  # Single head for simplicity

    # Initialize random tensors
    q = torch.randn(batch_size, num_queries, qk_dim)
    k = torch.randn(batch_size, num_keys, qk_dim)
    v = torch.randn(batch_size, num_keys, v_dim)
    key_padding_mask = torch.randint(
        0, 2, (batch_size, num_keys)
    ).bool()  # [batch_size, num_keys], True means to mask out

    # Create mask: [batch_size, num_queries, num_keys], broadcasted from key_padding_mask
    lookahead_mask = torch.randint(0, 2, (num_queries, num_keys)).bool()
    attention = DotProductAttention()

    # Initialize PyTorch's MultiheadAttention
    mha = torch.nn.MultiheadAttention(
        embed_dim=qk_dim,
        num_heads=num_heads,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
    )
    # Since nn.MultiheadAttention expects inputs as [seq_len, batch_size, embed_dim]
    q_mha = q.transpose(0, 1)  # [num_queries, batch_size, qk_dim]
    k_mha = k.transpose(0, 1)  # [num_keys, batch_size, qk_dim]
    v_mha = v.transpose(0, 1)  # [num_keys, batch_size, v_dim]

    # Initialize weights to identity matrices for comparison
    with torch.no_grad():
        # in_proj_weight shape: (3*embed_dim, embed_dim)
        proj_weight = torch.cat(
            [torch.eye(qk_dim), torch.eye(qk_dim), torch.eye(qk_dim)], dim=0
        )
        mha.in_proj_weight.copy_(proj_weight)
        # out_proj.weight shape: (embed_dim, embed_dim)
        mha.out_proj.weight.copy_(torch.eye(v_dim))
        # If there is an out_proj bias, set it to zero
        if mha.out_proj.bias is not None:
            mha.out_proj.bias.zero_()

    # Forward pass through MultiheadAttention
    output_mha, _ = mha(
        q_mha,
        k_mha,
        v_mha,
        attn_mask=lookahead_mask.clone(),
        key_padding_mask=key_padding_mask,
    )
    output_mha = output_mha.transpose(0, 1)  # [batch_size, num_queries, v_dim]
    output_custom = attention(
        q, k, v, attn_mask=lookahead_mask, key_padding_mask=key_padding_mask
    )
    # output_custom  = attention(q, k, v, attn_mask = mask)
    atol = 1e-4
    # TODO Remember to remove
    print(f"{output_custom}")
    print(f"{output_mha}")
    output_custom, output_mha = allclose_replace_nan(
        output_custom, output_mha, rtol=1e-05, atol=1e-08, sentinel=0.0
    )
    assert torch.allclose(
        output_custom, output_mha, atol=atol
    ), f"Comparative analysis failed: Outputs do not match. output_custom shape: {output_custom.shape}, output_mha shape: {output_mha.shape}"


def test_scaled_dot_product_gradient_flow():
    """
    Force backward() and ensures the source q, k, v's `.grad` attribute is set
    """
    batch_size = 1
    num_queries = 2
    num_keys = 3
    qk_dim = 4
    v_dim = 5

    # Create tensors with requires_grad=True
    q = torch.randn(batch_size, num_queries, qk_dim, requires_grad=True)
    k = torch.randn(batch_size, num_keys, qk_dim, requires_grad=True)
    v = torch.randn(batch_size, num_keys, v_dim, requires_grad=True)

    # Create mask
    mask = torch.zeros(batch_size, num_queries, num_keys)

    # Initialize attention module
    attention = DotProductAttention()

    # Forward pass
    output = attention(q, k, v, attn_mask=mask)

    # Define a simple loss (sum of all outputs)
    loss = output.sum()
    # Backward pass
    loss.backward()

    # Check that gradients are not None
    assert q.grad is not None, "Gradients not flowing for queries."
    assert k.grad is not None, "Gradients not flowing for keys."
    assert v.grad is not None, "Gradients not flowing for values."


def test_scaled_dot_product_numerical_precision():
    """
    Testing numerical stability of the scaled dot attention
    """
    batch_size = 1
    num_queries = 2
    num_keys = 2

    # Inputs with large values
    q = torch.tensor(
        [[[1e10, 0.0, 0.0, 0.0], [0.0, 1e10, 0.0, 0.0]]], dtype=torch.float32
    )
    k = torch.tensor(
        [[[1e10, 0.0, 0.0, 0.0], [0.0, 1e10, 0.0, 0.0]]], dtype=torch.float32
    )
    v = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32)

    mask = torch.zeros(batch_size, num_queries, num_keys)
    attention = DotProductAttention()
    output = attention(q, k, v, attn_mask=mask)
    # Check for NaNs or Infs
    assert not torch.isnan(output).any(), "Output contains NaNs."
    assert not torch.isinf(output).any(), "Output contains Infs."


def test_multi_head_attention():
    """
    Ultimate correctness check against the torch implementation with the same weight matrices
    """
    batch_size = 2
    num_queries = 4
    num_keys = 5
    qk_dim = 8
    v_dim = 8  # Set v_dim equal to qk_dim for comparison
    num_heads = 1  # Single head for simplicity

    # Initialize random tensors
    q = torch.randn(batch_size, num_queries, qk_dim)
    k = torch.randn(batch_size, num_keys, qk_dim)
    v = torch.randn(batch_size, num_keys, v_dim)
    # Since nn.MultiheadAttention expects inputs as [seq_len, batch_size, embed_dim]
    q_mha = q.transpose(0, 1)  # [num_queries, batch_size, qk_dim]
    k_mha = k.transpose(0, 1)  # [num_keys, batch_size, qk_dim]
    v_mha = v.transpose(0, 1)  # [num_keys, batch_size, v_dim]
    key_padding_mask = torch.randint(
        0, 2, (batch_size, num_keys)
    ).bool()  # [batch_size, num_keys], True means to mask out
    # Create mask: [batch_size, num_queries, num_keys], broadcasted from key_padding_mask
    lookahead_mask = torch.randint(0, 2, (num_queries, num_keys)).bool()
    attention = MultiHeadAttention(
        embed_dim=qk_dim,
        num_heads=num_heads,
    )
    output_custom = attention(
        q_mha, k_mha, v_mha, attn_mask=lookahead_mask, key_padding_mask=key_padding_mask
    )
    attention.eval()  # Disable dropout for testing
    mha = torch.nn.MultiheadAttention(
        embed_dim=qk_dim,
        num_heads=num_heads,
        bias=False,
        add_bias_kv=False,
        add_zero_attn=False,
    )
    mha.eval()  # Disable dropout for testing
    with torch.no_grad():
        # Concatenate weights for in_proj_weight
        in_proj_weight = torch.cat(
            [attention.Wq.weight, attention.Wk.weight, attention.Wv.weight], dim=0
        )
        mha.in_proj_weight.copy_(in_proj_weight)
        mha.out_proj.weight.copy_(attention.out_proj.weight)
    output_mha, _ = mha(
        q_mha,
        k_mha,
        v_mha,
        attn_mask=lookahead_mask,
        key_padding_mask=key_padding_mask,
    )
    atol = 1e-4
    output_custom, output_mha = allclose_replace_nan(
        output_custom, output_mha, rtol=1e-05, atol=1e-08, sentinel=0.0
    )
    assert torch.allclose(
        output_custom, output_mha, atol=atol
    ), f"Comparative analysis failed: Outputs do not match. output_custom shape: {output_custom.shape}, output_mha shape: {output_mha.shape}"
