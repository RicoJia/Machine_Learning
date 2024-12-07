# References:
# - https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# - Another example: https://iiosnail.blogspot.com/2024/10/nn-transfomer.html
# - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
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

BATCH_SIZE = 16
EMBEDDING_DIM = 16
NUM_HEADS = 8
DROPOUT_RATE = 0.1
MAX_SENTENCE_LENGTH = MAX_LENGTH
ENCODER_LAYER_NUM = 1
DECODER_LAYER_NUM = 1
NUM_EPOCHS = 800
GRADIENT_CLIPPED_NORM_MAX = 5.0
input_lang, output_lang, train_dataloader, pairs = get_dataloader(BATCH_SIZE)
INPUT_TOKEN_SIZE = input_lang.n_words
OUTPUT_TOKEN_SIZE = output_lang.n_words


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

        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=MAX_SENTENCE_LENGTH
        )
        self.encoder_embedding = nn.Embedding(num_tokens, dim_model)
        self.decoder_embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)
        nn.init.xavier_uniform_(self.encoder_embedding.weight)
        nn.init.xavier_uniform_(self.decoder_embedding.weight)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        """ Training Function of this Transformer wrapper

        Args: 
            src (_type_): (batch_size, src sequence length)
            tgt (_type_): (batch_size, tgt sequence length)
            tgt_mask (_type_, optional): (sequence length, sequence length)
            src_pad_mask (_type_, optional): (batch_size, src sequence length)
            tgt_pad_mask (_type_, optional): (batch_size, src sequence length)

        Returns:
            Logits: (batch_size, sequence length, num_tokens)
        """
        # Src size must be 
        # Tgt size must be 

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src_1, tgt_1 = src, tgt
        src = self.encoder_embedding(src) * math.sqrt(self.dim_model)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.dim_model)
        src_2, tgt_2 = src, tgt
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
        # (batch_size, sequence length, num_tokens)
        out = out.permute(1, 0, 2)

        if torch.isnan(out).any():
            print(f"NaN found in logits")
        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(
            mask == 0, -1e9
        )  # Convert zeros to small value. Not using -inf
        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return matrix == pad_token


def generate_square_subsequent_mask(sz, device):
    """
    EX for size=5:
    [[0., -inf, -inf, -inf, -inf],
     [0.,   0., -inf, -inf, -inf],
     [0.,   0.,   0., -inf, -inf],
     [0.,   0.,   0.,   0., -inf],
     [0.,   0.,   0.,   0.,   0.]]
    """
    mask = torch.triu(torch.ones(sz, sz, device=device) * -1e9, diagonal=1)
    return mask


def create_mask(src, tgt):
    tgt_seq_len = tgt.size(1)
    # 0 = unmask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
    src_padding_mask = src == PAD_token
    tgt_padding_mask = tgt == PAD_token
    return tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler(device=device, enabled=True)
    with torch.autograd.set_detect_anomaly(True):
        for src_batch, tgt_batch in dataloader:
            optimizer.zero_grad()  # is it before or after scaler?
            tgt_batch = tgt_batch.to(device)
            src_batch = src_batch.to(device)
            tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src_batch, tgt_batch
            )
            tgt_mask = tgt_mask.to(device=device)
            src_padding_mask = src_padding_mask.to(device=device)
            tgt_padding_mask = tgt_padding_mask.to(device=device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                logits = model(
                    src=src_batch,  # [30, 16], [max_length, batch size]
                    tgt=tgt_batch,  # [29, 16]
                    tgt_mask=tgt_mask,  # 99, 99
                    src_pad_mask=src_padding_mask,
                    tgt_pad_mask=tgt_padding_mask,
                )
                loss = criterion(
                    logits.reshape(
                        -1, logits.size(-1)
                    ),  # (batch_size * sequence length, output_token_dim)
                    tgt_batch.reshape(
                        -1
                    ),  # tgt_batch is (batch_size * sequence length)
                )

            # calculate gradients
            scaler.scale(loss).backward()
            for name, param in model.named_parameters():
                if torch.isinf(param.grad).any():
                    print("inf: ", name)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=GRADIENT_CLIPPED_NORM_MAX
                    )
                    print(
                        f"Applied gradient clipping to norm :{GRADIENT_CLIPPED_NORM_MAX}"
                    )
                if torch.isnan(param.grad).any():
                    print("nan: ", name)
            # unscale gradients (from float16 to float32)
            scaler.step(optimizer)
            # Adjusts the scaling factor for the next iteration. If gradients are too low, increase the scaling factor.
            scaler.update()
            # optimizer.step()  # this could be dangerous, because we are reapplying stale gradients?
            total_loss += loss.detach().item()
            if args.debug:
                training_logits_to_outuput_sentence(logits, tgt_batch, output_lang)
    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, epochs, start_epoch):
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

        print(f"Training loss: {train_loss:.4f}")
        print()
        save_model_and_optimizer(model, opt, epoch=epoch, path=MODEL_PATH)

    return train_loss_list, validation_loss_list


def training_logits_to_outuput_sentence(logits_batch, tgt_batch, output_lang):
    """ Translate all logits generated during training

    Args:
        logits_batch (_type_): [batch_size, sentence_length, ouput_token_dim]
        tgt_batch (_type_): [batch_size, sentence_length]
        output_lang (_type_): output_language dictionary object
    """
    for logits, tgt in zip(logits_batch, tgt_batch):
        # logits: [sentence_length, ouput_token_dim]
        ys = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
        _, indices = torch.max(logits, dim=-1)  # next_word:
        for i in range(MAX_SENTENCE_LENGTH):
            # pred has all timesteps, so does next_word
            next_word = indices[i].item()
            ys = torch.cat(
                [ys, torch.tensor([[next_word]], device=device)], dim=1
            )  # Shape: (1, tgt_seq_len + 1)
            # if next_word == EOS_token:
            #     break
        pred_tokens = ys.flatten()
        translated_tokens = []
        for token in pred_tokens:
            token_idx = token.item()
            word = output_lang.index2word.get(token_idx, "<unk>")
            translated_tokens.append(word)
        tgt_tokens = []
        for token in tgt:
            token_idx = token.item()
            word = output_lang.index2word.get(token_idx, "<unk>")
            tgt_tokens.append(word)
        print(
            f"Translated sentence during training: {translated_tokens}, target: {tgt_tokens}"
        )


# TODO: this might be broken, focus on training now
@torch.inference_mode()
def translate(model, src_sentence, output_lang):
    src_tokens = input_lang_sentence_to_tokens(
        src_sentence=src_sentence, input_lang=input_lang
    )
    src = torch.LongTensor([src_tokens]).to(device)

    ys = torch.tensor([[SOS_token]], dtype=torch.long, device=device)  # Shape: (1, 1)
    for i in range(MAX_SENTENCE_LENGTH):
        tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(
            device
        )  # Shape: (tgt_seq_len, tgt_seq_len)
        # Decode the target sequence
        logits = model(
            src=src,
            tgt=ys,
            # tgt_mask=tgt_mask
        )
        _, next_word = torch.max(logits, dim=-1)
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


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.autograd.set_detect_anomaly(args.debug)
    # input_vocab_dim, embedding_dim, num_heads
    model = Transformer(
        num_tokens=INPUT_TOKEN_SIZE,
        dim_model=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=ENCODER_LAYER_NUM,
        num_decoder_layers=DECODER_LAYER_NUM,
        dropout_p=0.1,
    ).to(device)
    # opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_token)
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
            epochs=NUM_EPOCHS,
            start_epoch=start_epoch,
        )
    test_sentences = ["Eres tú", "Eres mala.", "Eres grande.", "Estás triste."]
    model.eval()
    for test_sentence in test_sentences:
        translation = translate(model, test_sentence, output_lang)
        print(f"Test Sentence: {test_sentence}, Translation: {translation}")
