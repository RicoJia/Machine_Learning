import torch
from ricomodels.og_transformer.og_transformer_encoder import OGPositionalEncoder


def test_og_positional_encoder():
    batch_size = 2
    sentence_length = 10
    embedding_size = 16
    max_sentence_length = 50

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


test_og_positional_encoder()
