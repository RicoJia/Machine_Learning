#!/usr/bin/env python3
import math

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
    Return an upper triangle
    tensor([[False,  True,  True],
            [False, False,  True],
            [False, False, False]])
    """
    # diagonal = 0 is to include the diagonal items
    return (
        1 - torch.tril(torch.ones(sequence_length, sequence_length), diagonal=0)
    ).bool()


class DotProductAttention(torch.nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        Args:
            q (torch.Tensor): [batch_size, query_num, qk_dim] or [batch_size, head_num, query_num, qk_dim]
            k (torch.Tensor): [batch_size, kv_num, qk_dim] or [batch_size, head_num, kv_num, qk_dim]
            v (torch.Tensor): [batch_size, kv_num, v_dim] or [batch_size, head_num, kv_num, qk_dim]
            attn_mask (torch.Tensor): or look-ahead mask, [query_num, kv_num]. 1 means "mask out"
                Later, they are multiplied by large negative values -1e9. so values can be ignored in softmax.
            key_padding_mask (torch.Tensor): [batch_size, kv_num]. 1 means "mask out"
        Returns:
            attention: [batch_size, query_num, v_dim] or [batch_size, head_num, query_num, qk_dim]
        """
        q_kT_scaled = (q @ k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(k.shape[-1], dtype=torch.float32)
        )
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            q_kT_scaled.masked_fill_(attn_mask.bool(), float("-inf"))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1)
            if q_kT_scaled.ndim == 4:
                key_padding_mask = key_padding_mask.unsqueeze(2)
            # [batch_size, query_num, kv_num]
            q_kT_scaled = q_kT_scaled.masked_fill(
                key_padding_mask,
                float("-inf"),
            )
        attention_weight = torch.nn.functional.softmax(q_kT_scaled, dim=-1)
        attention = attention_weight @ v
        # TODO In this implementation, there's a drop out
        # https://ricojia.github.io/2022/03/27/deep-learning-attention-mechanism/#scaled-dot-product-luong-attention
        return attention


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        1. Linearly transform q, k, v so that they all have the same hidden dimension hidden_size
        2. Split q', k', v' into heads
        3. Each group of q, k, v go into DotProductAttention
        4. The concatenated head is transformed into a shorter embedding through a dense layer, Wo
        """
        # embed_dim is also qk_dim,
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"Embed_dim: {embed_dim} must be divisible by num_heads: {num_heads}"
        # Doing Wq, Wk, Wv. By default, v is also assumed to be of length embed_dim
        self.Wq = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        # self.Wo
        self.out_proj = torch.nn.Linear(
            embed_dim, embed_dim, bias=False
        )  # TODO: by default, o is also of embed_dim?
        self.attention = DotProductAttention()
        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads
        self.embedding_dim = embed_dim

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        """
        Args: ACHTUNG: THIS IS WEIRD because num_queries is at the front
        q (torch.Tensor): [num_queries, batch_size, qk_dim]
        k (torch.Tensor): [num_keys, batch_size, qk_dim]
        v (torch.Tensor): [num_keys, batch_size, v_dim]
        """
        num_queries, batch_size, _ = q.size()
        num_keys = k.size(0)
        q_proj = self.Wq(q)  # [num_queries, batch_size, embed_dim]
        k_proj = self.Wk(k)  # [num_keys, batch_size, embed_dim]
        v_proj = self.Wv(v)  # [num_keys, batch_size, embed_dim]
        # now, split them into num_heads. How to calculate heads in parallel?
        q = q_proj.view(num_queries, batch_size, self.num_heads, self.head_dim)
        k = k_proj.view(num_keys, batch_size, self.num_heads, self.head_dim)
        v = v_proj.view(num_keys, batch_size, self.num_heads, self.head_dim)

        # [batch, head_num, num_keys/num_queries, embed_dim]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        # [batch_size, head_num, query_num, head_embed_dim]
        attention = self.attention(
            q=q, k=k, v=v, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        # [query_num, batch_size, head_num, head_embed_dim]
        attention = attention.permute(2, 0, 1, 3).contiguous()  # TODO? .contiguous()
        attention = attention.view(num_queries, batch_size, self.embedding_dim)
        attention_output = self.out_proj(attention)
        return attention_output


class PositionwiseFFN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim) -> None:
        super().__init__()
        self.dense1 = torch.nn.LazyLinear(hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.LazyLinear(output_dim)

    def forward(self, X):
        # (batch size, number of time steps, output_dim).
        return self.dense2(self.relu(self.dense1(X)))


class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout_rate=0.1,
    ) -> None:
        super().__init__()
        # need dropout. The torch implementation already has it
        self.mha = MultiHeadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.ffn = PositionwiseFFN(hidden_dim=embedding_dim, output_dim=embedding_dim)
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, X, attn_mask, key_padding_mask):
        # Self attention (batch_size, input_seq_len, embedding_dim)
        self_attn_output = self.mha(
            X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        # apply dropout layer to the self-attention output (~1 line)
        self_attn_output = self.dropout1(
            self_attn_output,
        )
        # Applying Skip Connection
        mult_attn_out = self.layernorm1(
            X + self_attn_output
        )  # (batch_size, input_seq_len, embedding_dim)
        ffn_output = self.ffn(
            mult_attn_out
        )  # (batch_size, input_seq_len, embedding_dim)
        ffn_output = self.dropout2(ffn_output)
        # Applying Skip Connection
        encoder_layer_out = self.layernorm2(
            ffn_output + mult_attn_out
        )  # (batch_size, input_seq_len, embedding_dim)
        return encoder_layer_out


class Encoder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_vocab_dim,
        encoder_layer_num,
        num_heads,
        max_sentence_length,
        dropout_rate=0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.positional_encoder = OGPositionalEncoder(
            max_sentence_length=max_sentence_length, embedding_size=self.embedding_dim
        )
        self.embedding_converter = torch.nn.Embedding(
            num_embeddings=input_vocab_dim, embedding_dim=self.embedding_dim
        )
        self.dropout_pre_encoder = torch.nn.Dropout(p=dropout_rate)
        self.encoder_layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    embedding_dim=self.embedding_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(encoder_layer_num)
            ]
        )

    def forward(self, X, attn_mask, key_padding_mask):
        # X: [Batch_Size, Sentence_length]
        X = self.embedding_converter(
            X
        )  # X: [Batch_Size, Sentence_length, embedding_size]
        X /= math.sqrt(float(self.embedding_dim))
        X = self.positional_encoder(X)  # applies positional encoding in addition
        # TODO Remember to remove
        print(f"X.shape: {X.shape}")
        X = self.dropout_pre_encoder(X)
        for encoder_layer in self.encoder_layers:
            X = encoder_layer(X, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return X


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
    print(f"{padding_mask}")

    look_ahead_mask = create_look_ahead_mask(sequence_length=3)
    print(f"{look_ahead_mask}")
