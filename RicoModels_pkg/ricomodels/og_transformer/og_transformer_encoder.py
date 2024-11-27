#!/usr/bin/env python3
import torch
from matplotlib import pyplot as plt


class OGPositionalEncoder(torch.nn.Module):
    def __init__(self, max_sentence_length, embedding_size):
        super().__init__()
        pos = torch.arange(start=0, end=max_sentence_length).unsqueeze(1)
        two_j = torch.arange(start=0, end=embedding_size) // 2 * 2
        # shape: [max_sentence_length, embedding_size]
        angles = pos / (10000 ** (two_j / embedding_size))
        self.positional_embedding = torch.zeros(
            (1, max_sentence_length, embedding_size)
        )
        self.positional_embedding[:, :, 0::2] = torch.sin(angles[:, 0::2])
        self.positional_embedding[:, :, 1::2] = torch.cos(angles[:, 1::2])

    def forward(self, X):
        # X: [Batch_Size, Time (sentence length), Channels (embeddings)]
        sentence_length = X.shape[1]
        X += self.positional_embedding[:, :sentence_length, :].to(X.device)
        return X


def create_padding_mask(padded_token_ids):
    # We assume this has been trucated, then padded with 0 (for short sentences)
    # [Batch_size, Time]
    mask = (padded_token_ids != 0).float()
    return mask


def create_look_ahead_mask(sequence_length):
    """
    Return a lower triangle
    tensor([[ True, False, False],
            [ True,  True, False],
            [ True,  True,  True]])
    """
    # diagonal = 0 is to include the diagonal items
    return torch.tril(torch.ones(sequence_length, sequence_length), diagonal=0).bool()


# class PaddingMask(torch.nn.Module):
#     def __init__(self, max_sentence_length):
#         super().__init__()
#     def forward(self, X):


def _plot_positional_encoder(
    og_positional_encoder: OGPositionalEncoder, max_sentence_length, embedding_size
):
    plt.pcolormesh(og_positional_encoder.positional_embedding[0, :, :], cmap="RdBu")
    plt.xlabel("d")
    plt.xlim((0, embedding_size))
    plt.ylabel("Position")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    max_sentence_length = 100
    embedding_size = 128
    og_positional_encoder = OGPositionalEncoder(
        max_sentence_length=max_sentence_length, embedding_size=embedding_size
    )
    # _plot_positional_encoder(og_positional_encoder, max_sentence_length=max_sentence_length, embedding_size=embedding_size)

    # Sample input sequences with padding (batch_size, seq_len)
    input_seq = torch.tensor(
        [
            [5, 7, 9, 0, 0],  # Sequence 1 (padded)
            [3, 2, 4, 1, 0],  # Sequence 2 (padded)
            [6, 1, 8, 4, 2],  # Sequence 3 (no padding)
        ]
    )
    padding_mask = create_padding_mask(input_seq)
    # TODO Remember to remove
    print(f"{padding_mask}")
    # print(torch.nn.functional.softmax(input_seq + (1 - padding_mask) * -1e9))

    look_ahead_mask = create_look_ahead_mask(sequence_length=3)
    print(f"{look_ahead_mask}")
