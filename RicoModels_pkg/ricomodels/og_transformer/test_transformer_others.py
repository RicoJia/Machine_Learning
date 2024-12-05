import torch
import torch.nn as nn
import torch.optim as optim
import random
from ricomodels.seq2seq.dataload_seq2seq import (
    get_dataloader,
    EOS_token,
    SOS_token,
    MAX_LENGTH,
    PAD_token,
    input_lang_sentence_to_tokens,
)
from ricomodels.utils.data_loading import get_package_dir
from ricomodels.og_transformer.translator_transformer import (
    save_model_and_optimizer,
    MODEL_PATH,
    load_model_and_optimizer,
    parse_args,
)
import os
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]
        )


@torch.inference_mode()
def greedy_decode(model, src, src_key_padding_mask, max_len, start_symbol):

    for i in range(max_len):
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
        out = model.final_dense(out)  # Shape: (tgt_seq_len, 1, vocab_size)
        prob = out[-1, 0, :]  # Shape: (vocab_size)
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.item()
        ys = torch.cat(
            [ys, torch.tensor([[next_word]], device=device)], dim=1
        )  # Shape: (1, tgt_seq_len + 1)
        if next_word == EOS_token:
            break
    return ys


@torch.inference_mode()
def translate(model, src_sentence, output_lang):
    src_tokens = input_lang_sentence_to_tokens(
        src_sentence=src_sentence, input_lang=input_lang
    )
    src = torch.LongTensor([src_tokens]).to(device)

    ys = torch.tensor([[SOS_token]], dtype=torch.long, device=device)  # Shape: (1, 1)
    src_mask = (src == 0).transpose(0, 1).to(device)
    for i in range(MAX_SENTENCE_LENGTH):
        tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(
            device
        )  # Shape: (tgt_seq_len, tgt_seq_len)
        # Decode the target sequence
        pred = model(src=src, tgt=ys, tgt_mask=tgt_mask)
        _, next_word = torch.max(pred, dim=-1)
        # pred has all timesteps, so does next_word
        next_word = next_word[-1].item()
        ys = torch.cat(
            [ys, torch.tensor([[next_word]], device=device)], dim=1
        )  # Shape: (1, tgt_seq_len + 1)
        if next_word == EOS_token:
            break
    tgt_tokens = ys.flatten()
    translated_tokens = []
    for token in tgt_tokens:
        token_idx = token.item()
        word = output_lang.index2word.get(token_idx, "<unk>")
        translated_tokens.append(word)

    # Exclude SOS and EOS tokens
    if translated_tokens[0] == SOS_token:
        translated_tokens = translated_tokens[1:]
    if translated_tokens[-1] == EOS_token:
        translated_tokens = translated_tokens[:-1]

    # Join tokens into a single string
    translated_sentence = " ".join(translated_tokens)
    return translated_sentence


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=MAX_SENTENCE_LENGTH
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return matrix == pad_token


def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
    return mask


def create_mask(src, tgt):
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


def train_epoch(model, optimizer, criterion, dataloader):
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


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, start_epoch):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(start_epoch, epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss = train_epoch(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]

        # validation_loss = validation_loop(model, loss_fn, val_dataloader)
        # validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        # print(f"Validation loss: {validation_loss:.4f}")
        print()
        save_model_and_optimizer(model, opt, epoch=epoch, path=MODEL_PATH)

    return train_loss_list, validation_loss_list


BATCH_SIZE = 16
# NUM_KEYS = NUM_QUERIES = MAX_SENTENCE_LENGTH
EMBEDDING_DIM = 64
NUM_HEADS = 4
DROPOUT_RATE = 0.1
MAX_SENTENCE_LENGTH = MAX_LENGTH
ENCODER_LAYER_NUM = 3
DECODER_LAYER_NUM = 3
NUM_EPOCHS = 80
input_lang, output_lang, train_dataloader, pairs = get_dataloader(BATCH_SIZE)
INPUT_TOKEN_SIZE = input_lang.n_words
OUTPUT_TOKEN_SIZE = output_lang.n_words

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # input_vocab_dim, embedding_dim, num_heads
    model = Transformer(
        num_tokens=INPUT_TOKEN_SIZE,
        dim_model=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=ENCODER_LAYER_NUM,
        num_decoder_layers=DECODER_LAYER_NUM,
        dropout_p=0.1,
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model, opt, start_epoch = load_model_and_optimizer(
        model,
        opt,
        path=os.path.join(
            get_package_dir(),
            "og_transformer",
            MODEL_PATH,
        ),
        device=device,
    )
    if not args.eval:
        train_loss_list, validation_loss_list = fit(
            model,
            opt,
            loss_fn,
            train_dataloader,
            val_dataloader=None,
            epochs=NUM_EPOCHS,
            start_epoch=start_epoch,
        )
    test_sentences = ["Eres tú", "Eres mala.", "Eres grande.", "Estás triste."]
    model.eval()
    for test_sentence in test_sentences:
        translation = translate(model, test_sentence, output_lang)
        print(f"Test Sentence: {test_sentence}, Translation: {translation}")
