#!/usr/bin/env python3
import math

import torch
from torch import nn
from ricomodels.og_transformer.og_transformer import OGPositionalEncoder
from ricomodels.seq2seq.dataload_seq2seq import (
    get_dataloader,
    EOS_token,
    SOS_token,
    MAX_LENGTH,
    PAD_token,
)
from ricomodels.utils.training_tools import (
    load_model_and_optimizer,
    save_model_and_optimizer,
)
import os
import argparse

BATCH_SIZE = 16
# NUM_KEYS = NUM_QUERIES = MAX_SENTENCE_LENGTH
NUM_KEYS = 50
NUM_QUERIES = 50
EMBEDDING_DIM = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
MAX_SENTENCE_LENGTH = MAX_LENGTH
ENCODER_LAYER_NUM = 3
DECODER_LAYER_NUM = 3
NUM_EPOCHS = 31

input_lang, output_lang, train_dataloader, pairs = get_dataloader(BATCH_SIZE)
INPUT_TOKEN_SIZE = input_lang.n_words
OUTPUT_TOKEN_SIZE = output_lang.n_words
MODEL_PATH = "spanish_to_english.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", "-e", action="store_true", default=True)
    parser.add_argument("--debug", "-d", action="store_true", default=True)
    args = parser.parse_args()
    return args


@torch.inference_mode()
def greedy_decode(model, src, src_key_padding_mask, max_len, start_symbol):
    """
    Performs greedy decoding with the Transformer model.

    Args:
        model: The Transformer model.
        src: Tensor of shape (src_seq_len, 1), the source sequence indices.
        src_key_padding_mask: Tensor of shape (1, src_seq_len), mask for the src keys per batch.
        max_len: int, the maximum length of the generated sequence.
        start_symbol: int, the index of the start token in the vocabulary.

    Returns:
        Tensor of shape (1, tgt_seq_len), the generated target sequence indices.
    """
    model.eval()
    device = src.device
    src_emb = model.encoder_embedding(
        src.transpose(0, 1)
    )  # Shape: (1, src_seq_len, embedding_dim)
    src_emb = src_emb * math.sqrt(model.encoder_embedding.embedding_dim)
    src_emb = model.encoder_positional_encoding(src_emb)
    src_emb = model.encoder_dropout(src_emb)
    src_emb = src_emb.permute(1, 0, 2)  # Shape: (src_seq_len, 1, embedding_dim)

    memory = model.transformer.encoder(
        src_emb, src_key_padding_mask=src_key_padding_mask  # Shape: (1, src_seq_len)
    )  # Shape: (src_seq_len, 1, embedding_dim)
    # Initialize the target sequence with the start symbol
    ys = torch.tensor(
        [[start_symbol]], dtype=torch.long, device=device
    )  # Shape: (1, 1)
    for i in range(max_len - 1):
        # Embed the current target sequence
        tgt_emb = model.decoder_embedding(ys)  # Shape: (1, tgt_seq_len, embedding_dim)
        tgt_emb = tgt_emb * math.sqrt(model.decoder_embedding.embedding_dim)
        tgt_emb = model.decoder_positional_encoding(tgt_emb)
        tgt_emb = model.decoder_dropout(tgt_emb)
        tgt_emb = tgt_emb.permute(1, 0, 2)  # Shape: (tgt_seq_len, 1, embedding_dim)
        tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(
            device
        )  # Shape: (tgt_seq_len, tgt_seq_len)

        # Decode the target sequence
        out = model.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            # memory_key_padding_mask=src_key_padding_mask,  # Shape: (1, src_seq_len)
            # tgt_key_padding_mask=None
        )  # Shape: (tgt_seq_len, 1, embedding_dim)
        out = out.permute(1, 0, 2)  # [batch_size, tgt_seq_len, embedding_dim]
        # Project the decoder output to the vocabulary
        out = model.final_dense(out)  # Shape: (tgt_seq_len, 1, vocab_size)
        # Get the probabilities for the next token (from the last time step)
        prob = out[-1, 0, :]  # Shape: (vocab_size)
        # Select the token with the highest probability
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.item()
        # Append the predicted token to the target sequence
        ys = torch.cat(
            [ys, torch.tensor([[next_word]], device=device)], dim=1
        )  # Shape: (1, tgt_seq_len + 1)
        # #TODO Remember to remove
        # print(f'============================')
        # print(f'Rico: prob {prob}')
        # print(f'Rico: tgt {tgt_emb}')
        # print(f"Time step {i + 1}:")
        # print(f"  ys: {ys}")
        # print(f"  next_word: {next_word}")
        # print(f"  prob[max]: {prob[next_word].item()}")
        # Stop decoding if the end-of-sequence token is generated
        if next_word == EOS_token:
            break
    return ys


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
        # self.final_relu = torch.nn.ReLU()
        # self.final_softmax = torch.nn.Softmax(dim=-1)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        tgt_mask=None,
    ):
        """
        Args:
            src : [batch_size, src_seq_len]
            tgt [batch_size, tgt_seq_len]
            src_key_padding_mask (_type_): [batch_size, MAX_LENGTH]?
            tgt_key_padding_mask (_type_): [batch_size, MAX_LENGTH]?
            tgt_mask (_type_, optional): [NUM_KEYS, NUM_QUERIES]?
        """
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
            # TODO: to see if this is necessary
            tgt_mask=tgt_mask,
            # src_key_padding_mask=src_key_padding_mask,
            # tgt_key_padding_mask=tgt_key_padding_mask,
            # memory_key_padding_mask=src_key_padding_mask,
        )  # [tgt_seq_len, batch_size, embedding_dim]

        # # Final projection
        # TODO
        # dec_output = dec_output.permute(
        #     1, 0, 2
        # )  # [batch_size, tgt_seq_len, embedding_dim]
        logits = self.final_dense(
            dec_output
        )  # [batch_size, tgt_seq_len, target_vocab_dim]
        return logits


def generate_square_subsequent_mask(sz, device):
    # masked out items are -inf here. TODO:
    mask = torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
    return mask


def create_mask(src, tgt):
    """Create padding masks and attn masks for source and target

    Args:
        src (_type_): [batch, input_sentence_length]
        tgt (_type_): [batch, output_sentence_length]
    Outputs:
        tgt_mask: attn mask of the target.
    """
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)

    # 0 = unmask
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(
        torch.bool
    )
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)

    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, dataloader):
    model.train()
    total_loss = 0
    for src_batch, tgt_batch in dataloader:
        tgt_batch = tgt_batch.to(device)
        src_batch = src_batch.to(device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src_batch, tgt_batch
        )
        # src_batch = src_batch.transpose(0, 1)
        src_mask = src_mask.to(device=device)
        tgt_mask = tgt_mask.to(device=device)
        src_padding_mask = src_padding_mask.to(device=device)
        tgt_padding_mask = tgt_padding_mask.to(device=device)
        logits = model(
            src=src_batch,  # [30, 16], [max_length, batch size]
            tgt=tgt_batch,  # [29, 16]
            tgt_mask=tgt_mask,  # 99, 99
            # src_key_padding_mask = src_padding_mask, #batch, max_length
            # tgt_key_padding_mask = tgt_padding_mask, #batch, max_length-1 TODO
        )
        # tgt_out = tgt_batch[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_batch.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / len(dataloader)


def translate(model, src_sentence):
    src_tokens = [
        input_lang.word2index.get(word, 0) for word in src_sentence.split(" ")
    ]
    src = torch.LongTensor(src_tokens).view(-1, 1).to(device)
    src_mask = (src == 0).transpose(0, 1).to(device)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=MAX_SENTENCE_LENGTH, start_symbol=SOS_token
    ).flatten()
    translated_tokens = []
    for token in tgt_tokens:
        token_idx = token.item()
        word = output_lang.index2word.get(token_idx, "<unk>")  # TODO?
        translated_tokens.append(word)

    # Exclude SOS and EOS tokens
    if translated_tokens[0] == SOS_token:
        translated_tokens = translated_tokens[1:]
    if translated_tokens[-1] == EOS_token:
        translated_tokens = translated_tokens[:-1]

    # Join tokens into a single string
    translated_sentence = " ".join(translated_tokens)
    return translated_sentence


if __name__ == "__main__":
    args = parse_args()
    model = PyTorchTransformer(
        embedding_dim=EMBEDDING_DIM,
        input_vocab_dim=INPUT_TOKEN_SIZE,
        target_vocab_dim=OUTPUT_TOKEN_SIZE,
        num_layers=DECODER_LAYER_NUM,
        num_heads=NUM_HEADS,
        max_sentence_length=MAX_SENTENCE_LENGTH,
        dropout_rate=DROPOUT_RATE,
        dim_feedforward=EMBEDDING_DIM,
    ).to(
        device
    )  # Move model to device immediately

    # Define the loss function
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_token
    )  # TODO: REALLY? Assuming 0 is the padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model, optimizer, start_epoch = load_model_and_optimizer(
        model, optimizer, path=MODEL_PATH, device=device
    )
    if not args.eval:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            epoch_loss = train_epoch(model, optimizer, train_dataloader)
            save_model_and_optimizer(model, optimizer, epoch=epoch, path=MODEL_PATH)
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    test_sentences = ["Eres tú", "Eres mala.", "Eres grande.", "Estás triste."]
    for test_sentence in test_sentences:
        translation = translate(model, test_sentence)
        print(f"Test Sentence: {test_sentence}, Translation: {translation}")
