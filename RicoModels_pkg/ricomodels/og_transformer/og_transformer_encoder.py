#!/usr/bin/env python3
import torch


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


OGPositionalEncoder(max_sentence_length=4, embedding_size=16)
