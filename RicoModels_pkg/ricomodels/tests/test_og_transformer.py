import pytest
import torch
from ricomodels.og_transformer.og_transformer import (
    DotProductAttention,
    Encoder,
    EncoderLayer,
    MultiHeadAttention,
    OGPositionalEncoder,
    DecoderLayer,
    Decoder,
    Transformer,
)
from ricomodels.utils.predict_tools import allclose_replace_nan
import math


##################################################################################################
## Constants and Configs
##################################################################################################

# TODO: this can be re-orged
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


BATCH_SIZE = 16
# NUM_KEYS = NUM_QUERIES = MAX_SENTENCE_LENGTH
NUM_KEYS = 50
NUM_QUERIES = 50
EMBEDDING_DIM = 16
NUM_HEADS = 8
DROPOUT_RATE = 0.1
INPUT_TOKEN_SIZE = 100
OUTPUT_TOKEN_SIZE = 150
MAX_SENTENCE_LENGTH = 50
ENCODER_LAYER_NUM = 2
DECODER_LAYER_NUM = 2
torch.manual_seed(42)

##################################################################################################
## Positional Encoder, Scaled-Dot Attention, Multi-Head Attention
##################################################################################################


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
    output, attention_weight = attention(q, k, v, mask)
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
    output, attn_weight = attention(q, k, v, attn_mask=mask)
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
    attn_mask = torch.randint(0, 2, (num_queries, num_keys)).bool()
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
        attn_mask=attn_mask.clone(),
        key_padding_mask=key_padding_mask,
    )
    output_mha = output_mha.transpose(0, 1)  # [batch_size, num_queries, v_dim]
    output_custom, attn_weight = attention(
        q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask
    )
    # output_custom  = attention(q, k, v, attn_mask = mask)
    atol = 1e-4
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
    output, _ = attention(q, k, v, attn_mask=mask)

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
    output, _ = attention(q, k, v, attn_mask=mask)
    # Check for NaNs or Infs
    assert not torch.isnan(output).any(), "Output contains NaNs."
    assert not torch.isinf(output).any(), "Output contains Infs."


def copy_weights_multi_head_attn(my_mha, torch_mha):
    with torch.no_grad():
        in_proj_weight = torch.cat(
            [my_mha.Wq.weight, my_mha.Wk.weight, my_mha.Wv.weight], dim=0
        )
        torch_mha.in_proj_weight.copy_(in_proj_weight)
        torch_mha.out_proj.weight.copy_(my_mha.out_proj.weight)


def test_multi_head_attention():
    """
    Ultimate correctness check against the torch implementation with the same weight matrices
    """
    batch_size = 2
    num_queries = 4
    num_keys = 5
    qk_dim = 8
    v_dim = 8  # Set v_dim equal to qk_dim for comparison
    num_heads = 2

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
    attn_mask = torch.randint(0, 2, (num_queries, num_keys)).bool()
    attention = MultiHeadAttention(
        embed_dim=qk_dim,
        num_heads=num_heads,
    )
    output_custom, _ = attention(
        q_mha, k_mha, v_mha, attn_mask=attn_mask, key_padding_mask=key_padding_mask
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
    # Concatenate weights for in_proj_weight
    copy_weights_multi_head_attn(my_mha=attention, torch_mha=mha)
    output_mha, _ = mha(
        q_mha,
        k_mha,
        v_mha,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )
    atol = 1e-4
    output_custom, output_mha = allclose_replace_nan(
        output_custom, output_mha, rtol=1e-05, atol=1e-08, sentinel=0.0
    )
    assert torch.allclose(
        output_custom, output_mha, atol=atol
    ), f"Comparative analysis failed: Outputs do not match. output_custom shape: {output_custom.shape}, output_mha shape: {output_mha.shape}"


##################################################################################################
## Encoder Layer Tests
##################################################################################################


@pytest.fixture
def key_padding_mask():
    """As long as the attention_weight is longer than 2 elements,
    we should never have a row being completely masked out"""
    mask = torch.zeros(BATCH_SIZE, NUM_KEYS)
    mask[:, -1:] = 1
    return mask.bool()


@pytest.fixture
def attn_mask():
    """As long as the attention_weight is longer than 2 elements,
    we should never have a row being completely masked out"""
    mask = torch.zeros(NUM_QUERIES, NUM_KEYS).bool()
    mask[:, 0] = True
    return mask


@pytest.fixture
def encoder_layer():
    """Fixture to create an EncoderLayer instance."""
    return EncoderLayer(
        embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE
    )


@pytest.fixture
def input_tensor():
    """Fixture to create a random input tensor."""
    return torch.randn(NUM_QUERIES, BATCH_SIZE, EMBEDDING_DIM, requires_grad=True)


def test_output_shape(encoder_layer, input_tensor, attn_mask, key_padding_mask):
    """Test if the output shape matches the input shape."""
    output = encoder_layer(input_tensor, attn_mask, key_padding_mask)
    assert (
        output.shape == input_tensor.shape
    ), f"Expected output shape {input_tensor.shape}, got {output.shape}"


def test_gradient_flow_no_mask(encoder_layer, input_tensor):
    """Predecessor test of test_gradient_flow"""
    output = encoder_layer(input_tensor, attn_mask=None, key_padding_mask=None)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None, "Gradients did not flow back to the input."
    assert input_tensor.grad.abs().sum() > 0, "Gradients are zero."


def test_gradient_flow(encoder_layer, input_tensor, attn_mask, key_padding_mask):
    """Smart test by checking if there's non-zero gradient after backward pass"""
    output = encoder_layer(input_tensor, attn_mask, key_padding_mask)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None, "Gradients did not flow back to the input."
    assert input_tensor.grad.abs().sum() > 0, "Gradients are zero."


def test_dropout_active_in_train(
    encoder_layer, input_tensor, attn_mask, key_padding_mask
):
    """Very smart test by having letting the encoder layer go thru the same inputs"""
    encoder_layer.train()
    output1 = encoder_layer(input_tensor, attn_mask, key_padding_mask)
    output2 = encoder_layer(input_tensor, attn_mask, key_padding_mask)
    # Since dropout is stochastic, outputs should differ
    assert not torch.allclose(
        output1, output2
    ), "Dropout did not introduce randomness during training."


def test_residual_connections(encoder_layer, input_tensor, attn_mask, key_padding_mask):
    """Test that residual connections are properly adding the input to the sublayer outputs."""
    encoder_layer.eval()
    with torch.no_grad():
        output = encoder_layer(input_tensor, attn_mask, key_padding_mask)
        # Since residual connections are present, output should differ from sublayer outputs
        # Compare output to input to ensure residuals have been added
        difference = torch.abs(output - input_tensor).mean()
        assert (
            difference > 0
        ), "Residual connections might not be functioning correctly."


def test_mask_handling(encoder_layer, input_tensor, attn_mask, key_padding_mask):
    """Test that applying masks affects the output as expected."""
    # Without masks
    output_no_mask = encoder_layer(input_tensor, attn_mask=None, key_padding_mask=None)
    output_with_mask = encoder_layer(input_tensor, attn_mask, key_padding_mask)
    # The outputs should differ when masks are applied
    assert not torch.allclose(
        output_no_mask, output_with_mask
    ), "Masking does not affect the output as expected."


@pytest.mark.parametrize(
    "embedding_dim,num_heads,dropout_rate",
    [
        (32, 4, 0.1),
        (64, 8, 0.2),
        (128, 16, 0.3),
    ],
)
def test_parameter_variations(
    embedding_dim, num_heads, dropout_rate, attn_mask, key_padding_mask
):
    """Test EncoderLayer with different embedding dimensions, number of heads, and dropout rates."""
    input_tensor = torch.randn(
        NUM_QUERIES, BATCH_SIZE, embedding_dim, requires_grad=True
    )
    encoder_layer = EncoderLayer(
        embedding_dim=embedding_dim, num_heads=num_heads, dropout_rate=dropout_rate
    )
    output = encoder_layer(input_tensor, attn_mask, key_padding_mask)
    assert (
        output.shape == input_tensor.shape
    ), f"Mismatch in output shape for embedding_dim={embedding_dim}, num_heads={num_heads}, dropout_rate={dropout_rate}"


def copy_weights_linear_layer(my_layer, torch_layer):
    with torch.no_grad():
        torch_layer.weight.copy_(my_layer.weight)
        if my_layer.bias is not None:
            torch_layer.bias.copy_(my_layer.bias)


def copy_weights_layer_norm(my_layer, torch_layer):
    with torch.no_grad():
        my_layer.weight.copy_(torch_layer.weight)
        my_layer.bias.copy_(torch_layer.bias)


def copy_weights_encoder_layer(torch_encoder_layer, my_encoder_layer):
    # Copy self-attention weights
    copy_weights_multi_head_attn(
        my_mha=my_encoder_layer.mha, torch_mha=torch_encoder_layer.self_attn
    )
    # # Copy feed-forward network weights
    with torch.no_grad():
        copy_weights_linear_layer(
            my_layer=my_encoder_layer.ffn.dense1,
            torch_layer=torch_encoder_layer.linear1,
        )
        copy_weights_linear_layer(
            my_layer=my_encoder_layer.ffn.dense2,
            torch_layer=torch_encoder_layer.linear2,
        )
        copy_weights_layer_norm(
            my_layer=my_encoder_layer.layernorm1, torch_layer=torch_encoder_layer.norm1
        )
        copy_weights_layer_norm(
            my_layer=my_encoder_layer.layernorm2, torch_layer=torch_encoder_layer.norm2
        )


def test_encoder_layer(input_tensor, attn_mask, key_padding_mask):
    encoder_layer = EncoderLayer(
        embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE
    )
    built_in_encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=EMBEDDING_DIM,
        nhead=NUM_HEADS,
        dim_feedforward=EMBEDDING_DIM,
        dropout=DROPOUT_RATE,
    )
    copy_weights_encoder_layer(built_in_encoder_layer, encoder_layer)

    encoder_layer.eval()
    built_in_encoder_layer.eval()
    with torch.no_grad():
        my_out = encoder_layer(
            X=input_tensor, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        torch_out = built_in_encoder_layer(
            src=input_tensor, src_mask=attn_mask, src_key_padding_mask=key_padding_mask
        )
    assert torch.allclose(torch_out, my_out, atol=1e-6, rtol=1e-4)


##################################################################################################
## Encoder Tests
##################################################################################################


@pytest.fixture
def input_tokens():
    # seq_length = torch.randint(low=0, high=MAX_SENTENCE_LENGTH, size=(1,)).item()
    return torch.randint(
        low=0,
        high=INPUT_TOKEN_SIZE,
        size=(BATCH_SIZE, MAX_SENTENCE_LENGTH),
        dtype=torch.long,
    )


@pytest.fixture
def full_encoder():
    """Fixture to create an EncoderLayer instance."""
    return Encoder(
        input_vocab_dim=INPUT_TOKEN_SIZE,
        encoder_layer_num=2,
        max_sentence_length=MAX_SENTENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
    )


class TestableTorchEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.built_in_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBEDDING_DIM,
            dropout=DROPOUT_RATE,
        )
        self.torch_embedding = torch.nn.Embedding(
            num_embeddings=INPUT_TOKEN_SIZE, embedding_dim=EMBEDDING_DIM
        )
        self.torch_positional_encoding = OGPositionalEncoder(
            max_sentence_length=MAX_SENTENCE_LENGTH, embedding_size=EMBEDDING_DIM
        )
        self.torch_encoder = torch.nn.TransformerEncoder(
            encoder_layer=self.built_in_encoder_layer, num_layers=ENCODER_LAYER_NUM
        )
        self.torch_dropout = torch.nn.Dropout(p=DROPOUT_RATE)

    def copy_weights(self, custom_encoder):
        self.torch_embedding.weight.data.copy_(
            custom_encoder.embedding_converter.weight.data.clone()
        )
        self.torch_positional_encoding.positional_embedding.copy_(
            custom_encoder.positional_encoder.positional_embedding
        )
        for i in range(ENCODER_LAYER_NUM):
            torch_layer = self.torch_encoder.layers[i]
            custom_layer = custom_encoder.encoder_layers[i]
            copy_weights_encoder_layer(
                my_encoder_layer=custom_layer, torch_encoder_layer=torch_layer
            )

    def forward(self, X, key_padding_mask):
        torch_X = self.torch_embedding(X) * math.sqrt(EMBEDDING_DIM)
        torch_X = self.torch_positional_encoding(torch_X)
        torch_X = self.torch_dropout(torch_X)
        torch_X = torch_X.permute(1, 0, 2)  # (seq_length, batch_size, embedding_dim)
        torch_out = self.torch_encoder(
            src=torch_X, mask=None, src_key_padding_mask=key_padding_mask
        )
        torch_out = torch_out.permute(1, 0, 2)  # [batch_size, input_seq_len, qk_dim]
        return torch_out


def test_encoder_output_shape(full_encoder, input_tokens):
    """Test if the output shape matches the input shape."""
    output = full_encoder(input_tokens, enc_padding_mask=None)
    expected_shape = (BATCH_SIZE, MAX_SENTENCE_LENGTH, EMBEDDING_DIM)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"


def test_encoder(input_tensor, attn_mask, key_padding_mask):
    custom_encoder = Encoder(
        embedding_dim=EMBEDDING_DIM,
        input_vocab_dim=INPUT_TOKEN_SIZE,
        encoder_layer_num=ENCODER_LAYER_NUM,
        num_heads=NUM_HEADS,
        max_sentence_length=MAX_SENTENCE_LENGTH,
        dropout_rate=DROPOUT_RATE,
    )
    torch_encoder = TestableTorchEncoder()
    with torch.no_grad():
        torch_encoder.copy_weights(custom_encoder=custom_encoder)

    torch_encoder.eval()
    custom_encoder.eval()

    X = torch.randint(
        0, INPUT_TOKEN_SIZE, (BATCH_SIZE, MAX_SENTENCE_LENGTH)
    )  # Random input indices

    with torch.no_grad():
        torch_out = torch_encoder(X, key_padding_mask)
        custom_out = custom_encoder(X=X, enc_padding_mask=key_padding_mask)
    assert torch.allclose(torch_out, custom_out, atol=1e-6, rtol=1e-4)


##################################################################################################
## Decoder Tests
##################################################################################################


def copy_decoder_layer_weights(torch_layer, custom_layer):
    copy_weights_multi_head_attn(
        my_mha=custom_layer.mha1, torch_mha=torch_layer.self_attn
    )
    copy_weights_multi_head_attn(
        my_mha=custom_layer.mha2, torch_mha=torch_layer.multihead_attn
    )
    copy_weights_linear_layer(
        my_layer=custom_layer.ffn.dense1, torch_layer=torch_layer.linear1
    )
    copy_weights_linear_layer(
        my_layer=custom_layer.ffn.dense2, torch_layer=torch_layer.linear2
    )
    copy_weights_layer_norm(
        my_layer=custom_layer.layernorm1, torch_layer=torch_layer.norm1
    )
    copy_weights_layer_norm(
        my_layer=custom_layer.layernorm2, torch_layer=torch_layer.norm2
    )
    copy_weights_layer_norm(
        my_layer=custom_layer.layernorm3, torch_layer=torch_layer.norm3
    )


def test_decoder_layer(input_tensor, attn_mask, key_padding_mask):
    """In this test, we choose target_sequence and encoder_output to be input_tensor
    Which is a simplification. They might have different dimensions:
        (tgt_seq_length, batch_size, embedding_dim)
        (memory_seq_length, batch_size, embedding_dim)
    """
    torch_decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=EMBEDDING_DIM,
        nhead=NUM_HEADS,
        dim_feedforward=EMBEDDING_DIM,
        dropout=DROPOUT_RATE,
        activation="relu",
    )
    custom_decoder_layer = DecoderLayer(
        embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout_rate=DROPOUT_RATE
    )
    copy_decoder_layer_weights(
        torch_layer=torch_decoder_layer, custom_layer=custom_decoder_layer
    )
    torch_decoder_layer.eval()
    custom_decoder_layer.eval()

    with torch.no_grad():
        target_sequence = input_tensor
        enc_out = input_tensor.clone()
        torch_out = torch_decoder_layer(
            tgt=target_sequence,
            memory=enc_out,
            tgt_mask=attn_mask,
            memory_mask=None,
            tgt_key_padding_mask=key_padding_mask,  # for masking key padding
            memory_key_padding_mask=key_padding_mask,
        )
        (
            custom_out,
            decoder_self_attn_weight,
            decoder_encoder_attn_weight,
        ) = custom_decoder_layer(
            X=target_sequence,
            enc_output=enc_out,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
    assert torch.allclose(torch_out, custom_out, atol=1e-6, rtol=1e-4)


class PyTorchDecoder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        target_vocab_dim,
        decoder_layer_num,
        max_sentence_length,
        dropout_rate=0.1,
        dim_feedforward=2048,
    ):
        super(PyTorchDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(
            num_embeddings=target_vocab_dim, embedding_dim=embedding_dim
        )
        self.positional_encoding = OGPositionalEncoder(
            max_sentence_length, embedding_dim
        )
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="relu",
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=decoder_layer_num
        )

    def forward(self, X, enc_output, lookahead_mask, key_padding_mask):
        """
        Args:
            X: [batch_size, tgt_seq_length] (target token indices)
            enc_output: [memory_seq_length, batch_size, embedding_dim] (encoder outputs)
            lookahead_mask: [tgt_seq_length, tgt_seq_length] (causal mask)
            key_padding_mask: [batch_size, tgt_seq_length] (padding mask for target)
        Returns:
            Output tensor: [batch_size, tgt_seq_length, embedding_dim]
        """
        # Embedding
        X = self.embedding(X)  # [batch_size, tgt_seq_length, embedding_dim]
        X = X * math.sqrt(self.embedding_dim)
        X = self.positional_encoding(X)  # [batch_size, tgt_seq_length, embedding_dim]
        X = self.dropout(X)
        X = X.permute(1, 0, 2)  # [tgt_seq_length, batch_size, embedding_dim]
        output = self.transformer_decoder(
            tgt=X,
            memory=enc_output,
            tgt_mask=lookahead_mask,
            memory_mask=None,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=None,  # Assuming no padding in encoder outputs
        )
        output = output.permute(1, 0, 2)  # [batch_size, tgt_seq_length, embedding_dim]
        return output

    def copy_weights(self, custom_decoder):
        with torch.no_grad():
            for i in range(DECODER_LAYER_NUM):
                torch_layer = self.transformer_decoder.layers[i]
                custom_layer = custom_decoder.dec_layers[i]
                copy_decoder_layer_weights(torch_layer, custom_layer)

            self.embedding.weight.data.copy_(
                custom_decoder.embedding_converter.weight.data.clone()
            )
            self.positional_encoding.positional_embedding.copy_(
                custom_decoder.positional_encoder.positional_embedding
            )


def test_decoder(input_tokens, input_tensor, attn_mask, key_padding_mask):
    custom_decoder = Decoder(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        target_vocab_dim=OUTPUT_TOKEN_SIZE,
        decoder_layer_num=DECODER_LAYER_NUM,  # Number of decoder layers
        max_sentence_length=MAX_SENTENCE_LENGTH,
        dropout_rate=DROPOUT_RATE,
    )
    torch_decoder = PyTorchDecoder(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        target_vocab_dim=OUTPUT_TOKEN_SIZE,
        decoder_layer_num=DECODER_LAYER_NUM,  # Number of decoder layers
        max_sentence_length=MAX_SENTENCE_LENGTH,
        dropout_rate=DROPOUT_RATE,
        dim_feedforward=EMBEDDING_DIM,
    )

    torch_decoder.copy_weights(custom_decoder=custom_decoder)
    custom_decoder.eval()
    torch_decoder.eval()
    with torch.no_grad():
        target_sequence = input_tokens
        enc_out = input_tensor.clone()
        torch_out = torch_decoder(
            X=target_sequence,
            enc_output=enc_out,
            lookahead_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        # TODO: this is a small discrepancy with the torch implementation
        enc_out = input_tensor.clone().permute(1, 0, 2)
        custom_out, decoder_self_attns, decoder_encoder_attns = custom_decoder(
            X=target_sequence,
            enc_output=enc_out,
            lookahead_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
    assert torch.allclose(torch_out, custom_out, atol=1e-6, rtol=1e-4)


##################################################################################################
## Transformer Test
##################################################################################################


class PyTorchTransformer(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_vocab_dim,
        target_vocab_dim,
        num_layers,
        num_heads,
        max_sentence_length,
        dropout_rate=0.1,
        dim_feedforward=2048,
    ):
        super(PyTorchTransformer, self).__init__()
        self.encoder_embedding = torch.nn.Embedding(
            num_embeddings=input_vocab_dim, embedding_dim=embedding_dim
        )
        self.decoder_embedding = torch.nn.Embedding(
            num_embeddings=target_vocab_dim, embedding_dim=embedding_dim
        )
        self.encoder_positional_encoding = OGPositionalEncoder(
            max_sentence_length, embedding_dim
        )
        self.decoder_positional_encoding = OGPositionalEncoder(
            max_sentence_length, embedding_dim
        )
        self.encoder_dropout = torch.nn.Dropout(p=dropout_rate)
        self.decoder_dropout = torch.nn.Dropout(p=dropout_rate)
        self.transformer = torch.nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="relu",
        )
        self.final_dense = torch.nn.Linear(embedding_dim, target_vocab_dim, bias=False)
        self.final_relu = torch.nn.ReLU()
        self.final_softmax = torch.nn.Softmax(dim=-1)

    def copy_transformer_weights(self, custom_model):
        with torch.no_grad():
            self.encoder_embedding.weight.data.copy_(
                custom_model.encoder.embedding_converter.weight.data.clone()
            )
            self.encoder_positional_encoding.positional_embedding.copy_(
                custom_model.encoder.positional_encoder.positional_embedding
            )
            for i in range(len(custom_model.encoder.encoder_layers)):
                torch_enc_layer = self.transformer.encoder.layers[i]
                custom_enc_layer = custom_model.encoder.encoder_layers[i]
                copy_weights_multi_head_attn(
                    torch_mha=torch_enc_layer.self_attn, my_mha=custom_enc_layer.mha
                )
                copy_weights_linear_layer(
                    torch_layer=torch_enc_layer.linear1,
                    my_layer=custom_enc_layer.ffn.dense1,
                )
                copy_weights_linear_layer(
                    torch_layer=torch_enc_layer.linear2,
                    my_layer=custom_enc_layer.ffn.dense2,
                )
                copy_weights_layer_norm(
                    torch_layer=torch_enc_layer.norm1,
                    my_layer=custom_enc_layer.layernorm1,
                )
                copy_weights_layer_norm(
                    torch_layer=torch_enc_layer.norm2,
                    my_layer=custom_enc_layer.layernorm2,
                )
            self.decoder_embedding.weight.data.copy_(
                custom_model.decoder.embedding_converter.weight.data.clone()
            )
            self.decoder_positional_encoding.positional_embedding.copy_(
                custom_model.decoder.positional_encoder.positional_embedding
            )
            for i in range(len(custom_model.decoder.dec_layers)):
                torch_dec_layer = self.transformer.decoder.layers[i]
                custom_dec_layer = custom_model.decoder.dec_layers[i]
                copy_weights_multi_head_attn(
                    torch_mha=torch_dec_layer.self_attn, my_mha=custom_dec_layer.mha1
                )
                copy_weights_multi_head_attn(
                    torch_mha=torch_dec_layer.multihead_attn,
                    my_mha=custom_dec_layer.mha2,
                )
                copy_weights_linear_layer(
                    torch_layer=torch_dec_layer.linear1,
                    my_layer=custom_dec_layer.ffn.dense1,
                )
                copy_weights_linear_layer(
                    torch_layer=torch_dec_layer.linear2,
                    my_layer=custom_dec_layer.ffn.dense2,
                )
                copy_weights_layer_norm(
                    torch_layer=torch_dec_layer.norm1,
                    my_layer=custom_dec_layer.layernorm1,
                )
                copy_weights_layer_norm(
                    torch_layer=torch_dec_layer.norm2,
                    my_layer=custom_dec_layer.layernorm2,
                )
            copy_weights_linear_layer(
                torch_layer=self.final_dense, my_layer=custom_model.final_dense_layer
            )

    def forward(
        self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask=None
    ):
        src_emb = self.encoder_embedding(
            src
        )  # [batch_size, src_seq_len, embedding_dim]
        src_emb = src_emb * math.sqrt(self.encoder_embedding.embedding_dim)

        src_emb = self.encoder_positional_encoding(src_emb)
        src_emb = self.encoder_dropout(src_emb)
        src_emb = src_emb.permute(1, 0, 2)  # [src_seq_len, batch_size, embedding_dim]

        # Embedding and positional encoding for decoder
        tgt_emb = self.decoder_embedding(
            tgt
        )  # [batch_size, tgt_seq_len, embedding_dim]
        tgt_emb = tgt_emb * math.sqrt(self.decoder_embedding.embedding_dim)
        tgt_emb = self.decoder_positional_encoding(tgt_emb)
        tgt_emb = self.decoder_dropout(tgt_emb)
        tgt_emb = tgt_emb.permute(1, 0, 2)  # [tgt_seq_len, batch_size, embedding_dim]

        # Transformer
        dec_output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # [tgt_seq_len, batch_size, embedding_dim]

        # Final projection
        dec_output = dec_output.permute(
            1, 0, 2
        )  # [batch_size, tgt_seq_len, embedding_dim]
        logits = self.final_dense(
            dec_output
        )  # [batch_size, tgt_seq_len, target_vocab_dim]
        # logits = self.final_relu(logits)
        # logits = self.final_softmax(logits)
        return logits


def test_transformer_against_pytorch(
    input_tokens, input_tensor, attn_mask, key_padding_mask
):
    # Instantiate PyTorch's Equivalent Transformer
    torch_transformer = PyTorchTransformer(
        embedding_dim=EMBEDDING_DIM,
        input_vocab_dim=INPUT_TOKEN_SIZE,
        target_vocab_dim=OUTPUT_TOKEN_SIZE,
        num_layers=ENCODER_LAYER_NUM,
        num_heads=NUM_HEADS,
        max_sentence_length=MAX_SENTENCE_LENGTH,
        dropout_rate=DROPOUT_RATE,
        dim_feedforward=EMBEDDING_DIM,
    )
    # Instantiate Your Custom Transformer
    custom_transformer = Transformer(
        embedding_dim=EMBEDDING_DIM,
        input_vocab_dim=INPUT_TOKEN_SIZE,
        target_vocab_dim=OUTPUT_TOKEN_SIZE,
        layer_num=ENCODER_LAYER_NUM,
        num_heads=NUM_HEADS,
        max_sentence_length=MAX_SENTENCE_LENGTH,
        dropout_rate=DROPOUT_RATE,
    )
    torch_transformer.copy_transformer_weights(custom_model=custom_transformer)
    torch_transformer.eval()
    custom_transformer.eval()
    # Forward Pass Through PyTorch's Transformer
    with torch.no_grad():
        torch_logits = torch_transformer(
            src=input_tokens,  # [batch_size, src_seq_length]
            tgt=input_tokens,  # [batch_size, tgt_seq_length]
            src_key_padding_mask=key_padding_mask,
            tgt_key_padding_mask=key_padding_mask,
            tgt_mask=attn_mask,
        )  # [batch_size, tgt_seq_length, target_vocab_dim]
        custom_logits, decoder_self_attns, decoder_encoder_attns = custom_transformer(
            input_sentences=input_tokens,  # [batch_size, src_seq_length]
            output_sentences=input_tokens,  # [batch_size, tgt_seq_length]
            enc_padding_mask=key_padding_mask,  # [batch_size, src_seq_length]
            attn_mask=attn_mask,  # [tgt_seq_length, tgt_seq_length]
            dec_padding_mask=key_padding_mask,  # [batch_size, tgt_seq_length]
        )  # [batch_size, tgt_seq_length, target_vocab_dim]
    assert torch.allclose(torch_logits, custom_logits, atol=1e-6, rtol=1e-4)
