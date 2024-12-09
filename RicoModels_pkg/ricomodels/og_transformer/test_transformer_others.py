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
from ricomodels.utils.training_tools import get_scheduled_probability, clip_gradients
import os
import math
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 4
EMBEDDING_DIM = 32
NUM_HEADS = 8
DROPOUT_RATE = 0.1
MAX_SENTENCE_LENGTH = MAX_LENGTH
ENCODER_LAYER_NUM = 4
DECODER_LAYER_NUM = 4
NUM_EPOCHS = 70
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
        """Training Function of this Transformer wrapper

        Args:
            src (_type_): (batch_size, src sequence length)
            tgt (_type_): (batch_size, tgt sequence length)
            tgt_mask (_type_, optional): (sequence length, sequence length)
            src_pad_mask (_type_, optional): (batch_size, src sequence length)
            tgt_pad_mask (_type_, optional): (batch_size, src sequence length)

        Returns:
            Logits: (batch_size, sequence length, num_tokens)
        """
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.encoder_embedding(src) * math.sqrt(self.dim_model)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.dim_model)
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


def generate_square_subsequent_mask(sz, device):
    """
    EX for size=5:
    [[0., -inf, -inf, -inf, -inf],
     [0.,   0., -inf, -inf, -inf],
     [0.,   0.,   0., -inf, -inf],
     [0.,   0.,   0.,   0., -inf],
     [0.,   0.,   0.,   0.,   0.]]
    """
    mask = torch.triu(torch.ones(sz, sz, device=device) * -1e9, diagonal=1).bool()
    return mask


def create_mask(src, tgt):
    # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
    # [False, False, False, True, True, True]
    src_padding_mask = src == PAD_token
    if tgt is not None:
        tgt_padding_mask = tgt == PAD_token
        tgt_seq_len = tgt.size(1)
        # 0 = unmask
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
    else:
        tgt_padding_mask = None
        tgt_mask = None
    return tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, criterion, dataloader, teacher_forcing_ratio):
    """Training Epoch with Mixed Precision and teacher forcing.

    Args:
        teacher_forcing_ratio (float): percentage of groundtruth being used.
            1.0 means using 100% groundtruth.

    Returns:
        Average loss
    """
    total_loss = 0
    scaler = torch.amp.GradScaler(device=device, enabled=True)
    model.train()
    with torch.autograd.set_detect_anomaly(args.debug):
        with tqdm(
            total=len(dataloader), desc=f"Training Progress", unit="batch"
        ) as pbar:
            for src_batch, tgt_batch in dataloader:
                optimizer.zero_grad()  # is it before or after scaler?
                tgt_batch = tgt_batch.to(device)
                src_batch = src_batch.to(device)
                tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src_batch, tgt_batch
                )
                # TODO
                # batch_size = tgt_batch.size(0)
                # decoder input is initialized to [<SOS>, <PAD> ...]
                # decoder_input = PAD_token * torch.ones(
                #     batch_size, MAX_SENTENCE_LENGTH, dtype=tgt_batch.dtype, device=device
                # )
                decoder_input = (
                    tgt_batch.clone().detach()
                )  # [batch_size, 1], a bunch of <SOS>
                decoder_input = decoder_input.to(device)
                last_output = None
                all_logits = []
                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=True
                ):
                    for t in range(1, MAX_SENTENCE_LENGTH):
                        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                            src_batch, decoder_input
                        )
                        tgt_mask = tgt_mask.to(device=device)
                        src_padding_mask = src_padding_mask.to(device=device)
                        tgt_padding_mask = tgt_padding_mask.to(device=device)

                        if (
                            last_output is not None
                            and random.random() > teacher_forcing_ratio
                        ):
                            # TODO: without this clone(), the embedding layer would expect a version lower than this
                            # I still don't understand why. Have tried: using clone(), with torch.no_grad(), setting decoder_input.requires_grad = False
                            # So let's keep it for now
                            decoder_input = decoder_input.clone()
                            decoder_input[:, t] = last_output.detach()

                        # [batch size, t, output_vocab_dim]
                        logits = model(
                            src=src_batch,  # [batch size, max_length]
                            tgt=decoder_input,  # [batch size, t]
                            tgt_mask=tgt_mask,  # 99, 99
                            src_pad_mask=src_padding_mask,
                            tgt_pad_mask=tgt_padding_mask,
                        )
                        last_logits = logits[:, -1, :]  # [batch_size, output_vocab_dim]
                        last_output = last_logits.argmax(
                            -1
                        )  # this won't change decoder_input's version
                        all_logits.append(
                            last_logits
                        )  # length: max_sentence_length - 1

                    # [batch_size, max_length, output_vocab_dim]
                    all_logits_tensor = torch.stack(all_logits, dim=1)
                    loss = criterion(
                        all_logits_tensor.reshape(
                            -1, logits.size(-1)
                        ),  # (batch_size * sequence length, output_token_dim)
                        tgt_batch[:, 1:].reshape(
                            -1
                        ),  # tgt_batch is ((batch_size-1) * sequence length). Need to take the first one out because we are not considering SOS now
                    )

                # calculate gradients
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradients(model, GRADIENT_CLIPPED_NORM_MAX)
                # unscale gradients (from float16 to float32)
                scaler.step(optimizer)
                # Adjusts the scaling factor for the next iteration. If gradients are too low, increase the scaling factor.
                scaler.update()
                # optimizer.step()  # this could be dangerous, because we are reapplying stale gradients?
                total_loss += loss.detach().item()

                pbar.update(1)

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

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            train_loss = train_epoch(
                model, opt, loss_fn, train_dataloader, teacher_forcing_ratio=0.8
            )
            prof.step()
        train_loss_list += [train_loss]

        print(f"Training loss: {train_loss:.4f}")
        print()
        save_model_and_optimizer(model, opt, epoch=epoch, path=MODEL_PATH)

    return train_loss_list, validation_loss_list


###############################################################################
# Validation Functions
###############################################################################


def test_decoder_translation(src_batch):
    """Takes in a batch of sentences in tokens, prints the translation

    Args:
        src_batch (_type_): _description_
    """
    model.eval()
    test_input_sentence_tokens = src_batch[0][1:-1]
    sentence = []
    for token in test_input_sentence_tokens:
        token_idx = token.item()
        if token_idx == EOS_token:
            break
        word = input_lang.index2word.get(token_idx, "<unk>")
        sentence.append(word)
    input_sentence = " ".join(sentence)
    translate(model, input_sentence, output_lang)


def training_logits_to_outuput_sentence(logits_batch, tgt_batch, output_lang):
    """Translate all logits generated during training

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
    src = PAD_token * torch.ones(
        (1, MAX_SENTENCE_LENGTH), dtype=torch.long, device=device
    )  # Shape: (1, 1)
    src[0, : len(src_tokens)] = torch.tensor(src_tokens, dtype=torch.long)
    src = src.to(device)

    ys = PAD_token * torch.ones(
        (1, MAX_SENTENCE_LENGTH), dtype=torch.long, device=device
    )  # Shape: (1, 1)
    ys[0][0] = SOS_token
    ys = ys.to(device)
    for i in range(MAX_SENTENCE_LENGTH - 1):
        # Only pass the portion of ys that we have generated so far
        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, ys)
        tgt_mask = tgt_mask.to(device=device)
        src_padding_mask = src_padding_mask.to(device=device)
        tgt_padding_mask = tgt_padding_mask.to(device=device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            # Run the model with the truncated ys
            logits = model(
                src=src,
                tgt=ys,
                tgt_mask=tgt_mask,
                src_pad_mask=src_padding_mask,
                tgt_pad_mask=tgt_padding_mask,  # Try disabling this at inference
            )
        # Get the next token prediction
        indices = torch.argmax(logits, dim=-1)
        last_word = indices[0, -1]  # The newly predicted token is the last one
        # Append the newly predicted token to ys
        ys[0, i + 1] = last_word
        if last_word == EOS_token:
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
    test_sentences = [
        "Eres tú",
        "Eres mala.",
        "Eres grande.",
        "Estás triste.",
        "estoy levantado",
        "soy tom",
        "soy gorda",
        "estoy en forma",
    ]
    model.eval()
    for test_sentence in test_sentences:
        translation = translate(model, test_sentence, output_lang)
        print(f"Test Sentence: {test_sentence}, Translation: {translation}")
