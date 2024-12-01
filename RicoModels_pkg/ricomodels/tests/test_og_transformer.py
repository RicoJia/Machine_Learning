import pytest
import torch
from ricomodels.og_transformer.og_transformer_encoder import (
    DotProductAttention,
    Encoder,
    EncoderLayer,
    MultiHeadAttention,
    OGPositionalEncoder,
)
from ricomodels.utils.predict_tools import allclose_replace_nan


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
    output_custom = attention(
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
    output_custom = attention(
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
BATCH_SIZE = 16
NUM_KEYS = 4
NUM_QUERIES = 4
EMBEDDING_DIM = 16
NUM_HEADS = 8
DROPOUT_RATE = 0.1
INPUT_TOKEN_SIZE = 100
MAX_SENTENCE_LENGTH = 50


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
    # TODO Remember to remove
    print(f"input_tensor.grad.abs().sum(): {input_tensor.grad.abs().sum()}")
    assert input_tensor.grad.abs().sum() > 0, "Gradients are zero."


def test_gradient_flow(encoder_layer, input_tensor, attn_mask, key_padding_mask):
    """Smart test by checking if there's non-zero gradient after backward pass"""
    output = encoder_layer(input_tensor, attn_mask, key_padding_mask)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None, "Gradients did not flow back to the input."
    # TODO Remember to remove
    print(f"input_tensor.grad.abs().sum(): {input_tensor.grad.abs().sum()}")
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


def test_encoder_output_shape(full_encoder, input_tokens):
    """Test if the output shape matches the input shape."""
    output = full_encoder(input_tokens)
    # TODO Remember to remove
    print(f"Rico: {output.shape}")
    expected_shape = (BATCH_SIZE, MAX_SENTENCE_LENGTH, EMBEDDING_DIM)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"


# def test_encoder():
#     encoder = torch.nn.TransformerEncoderLayer(
#             d_model=embedding_dim,
#             nhead=num_heads,
#             dim_feedforward=ff_dim,
#             batch_first=True
#         )
# def full_transformer_test():
#     """Fixture to create a random input tensor."""
#     sentences = [
#             # enc_input           dec_input         dec_output
#             ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
#             ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
#     ]
#     # Padding Should be Zero
#     source_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
#     source_vocab_size = len(source_vocab)
#     target_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
#     idx2word = {i: w for i, w in enumerate(target_vocab)}
#     # What does tokens look like? NUM_QUERIES?
#     source_len = 5 # max length of input sequence
#     target_len = 6

#     def make_data(sentences):
#         encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
#         for i in range(len(sentences)):
#             encoder_input = [source_vocab[word] for word in sentences[i][0].split()]
#             decoder_input = [target_vocab[word] for word in sentences[i][1].split()]
#             decoder_output = [target_vocab[word] for word in sentences[i][2].split()]
#             encoder_inputs.append(encoder_input)
#             decoder_inputs.append(decoder_input)
#             decoder_outputs.append(decoder_output)

#         return torch.LongTensor(encoder_inputs), torch.LongTensor(decoder_inputs), torch.LongTensor(decoder_outputs)

#     class Seq2SeqDataset(torch.utils.data.Dataset):
#         def __init__(self, encoder_input, decoder_input, decoder_output):
#             super(Seq2SeqDataset, self).__init__()
#             self.encoder_input = encoder_input
#             self.decoder_input = decoder_input
#             self.decoder_output = decoder_output

#         def __len__(self):
#             return self.encoder_input.shape[0]

#         def __getitem__(self, idx):
#             return self.encoder_input[idx], self.decoder_input[idx], self.decoder_output[idx]
