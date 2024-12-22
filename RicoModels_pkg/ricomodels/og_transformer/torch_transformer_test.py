# References:
# - https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# - Another example: https://iiosnail.blogspot.com/2024/10/nn-transfomer.html
# - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
from ricomodels.seq2seq.dataload_seq2seq import (
    get_dataloader,
    Lang,
    EOS_token,
    SOS_token,
    MAX_LENGTH,
    PAD_token,
    input_lang_sentence_to_tokens,
)
from ricomodels.utils.data_loading import get_package_dir
from ricomodels.og_transformer.translator_transformer import (
    save_model_and_optimizer,
    load_model_and_optimizer,
)
from ricomodels.utils.training_tools import get_scheduled_probability, clip_gradients
import os
import math
import numpy as np
from tqdm import tqdm
import wandb
from ricomodels.utils.visualization import TrainingTimer, model_summary
from typing import List

EFFECTIVE_BATCH_SIZE = 256
BATCH_SIZE = 64
ACCUMULATION_STEPS = int(EFFECTIVE_BATCH_SIZE / BATCH_SIZE)
EMBEDDING_DIM = 64
NUM_HEADS = 8
DROPOUT_RATE = 0.1
MAX_SENTENCE_LENGTH = MAX_LENGTH
ENCODER_DECODER_LAYER_NUM = 3
NUM_EPOCHS = 900
GRADIENT_CLIPPED_NORM_MAX = 5.0
TEACHER_FORCING_RATIO_MIN = 0.0
input_lang, output_lang, train_dataloader, pairs = get_dataloader(BATCH_SIZE)
INPUT_TOKEN_SIZE = input_lang.n_words
OUTPUT_TOKEN_SIZE = output_lang.n_words
# Not recommended, because in certain predictions, EOS might yield the same loss as wrong predictions.
TERMINATE_TRAINING_UPON_EOS = False
MODEL_PATH = os.path.join(
    get_package_dir(),
    "og_transformer",
    f"spanish2english_{MAX_SENTENCE_LENGTH}tokens_{EMBEDDING_DIM}dim_{ENCODER_DECODER_LAYER_NUM}layers_{INPUT_TOKEN_SIZE}i.pth",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", "-e", action="store_true", default=False)
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--custom", "-c", action="store_true", default=False)
    parser.add_argument("--summary", "-s", action="store_true", default=False)
    args = parser.parse_args()
    return args


def generate_square_subsequent_mask(sz):
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


def create_mask_on_device(src, tgt, device):
    """Create masks optionally on the current chosen device

    Args:
        src (_type_): _description_
        tgt (_type_): _description_
        device (_type_): _description_

    Returns:
        tgt_mask (optional) look ahead mask
        src_padding_mask: padding mask that masks out padding in src
        tgt_padding_mask (optional) padding mask if tgt is specified
    """
    # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
    # [False, False, False, True, True, True]
    src_padding_mask = (src == PAD_token).to(device)
    if tgt is not None:
        tgt_padding_mask = (tgt == PAD_token).to(device)
        tgt_seq_len = tgt.size(1)
        # 0 = unmask
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    else:
        tgt_padding_mask = None
        tgt_mask = None
    return tgt_mask, src_padding_mask, tgt_padding_mask


class OGPositionalEncoder(torch.nn.Module):
    def __init__(self, max_sentence_length, embedding_size, dropout=0.1):
        super().__init__()
        pos = torch.arange(
            start=0, end=max_sentence_length, dtype=torch.float
        ).unsqueeze(1)
        two_j = torch.arange(start=0, end=embedding_size, dtype=torch.float) // 2 * 2
        # shape: [max_sentence_length, embedding_size]
        angles = pos / (10000 ** (two_j / embedding_size))
        positional_embedding = torch.zeros((1, max_sentence_length, embedding_size))
        positional_embedding[:, :, 0::2] = torch.sin(angles[:, 0::2])
        positional_embedding[:, :, 1::2] = torch.cos(angles[:, 1::2])
        self.dropout = torch.nn.Dropout(p=dropout)
        self.register_buffer("positional_embedding", positional_embedding)

    def forward(self, X):
        # X: [Batch_Size, Time (sentence length), Channels (embeddings)]
        sentence_length = X.shape[1]
        X += self.positional_embedding[:, :sentence_length, :]
        X = self.dropout(X)
        return X


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    def __init__(
        self,
        input_token_size,
        output_token_size,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.positional_encoder = OGPositionalEncoder(
            max_sentence_length=MAX_SENTENCE_LENGTH,
            embedding_size=dim_model,
            dropout=dropout_p,
        )
        self.encoder_embedding = nn.Embedding(input_token_size, dim_model)
        nn.init.xavier_uniform_(self.encoder_embedding.weight)
        self.decoder_embedding = nn.Embedding(output_token_size, dim_model)
        nn.init.xavier_uniform_(self.decoder_embedding.weight)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, output_token_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ):
        """Training Function of this Transformer wrapper

        Args:
            src (torch.Tensor): (batch_size, src sequence length)
            tgt (torch.Tensor): (batch_size, tgt sequence length)
            tgt_mask (torch.Tensor, optional): (sequence length, sequence length)
            src_pad_mask (torch.Tensor, optional): (batch_size, src sequence length)
            tgt_pad_mask (torch.Tensor, optional): (batch_size, src sequence length)

        Returns:
            Logits: (batch_size, sequence length, num_tokens)
        """
        tgt_mask, src_pad_mask, tgt_pad_mask = create_mask_on_device(
            src, tgt, device=device
        )
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


###############################################################################
# Training Functions
###############################################################################


def train_with_teacher_enforcing(
    src_batch: torch.Tensor,
    tgt_batch: torch.Tensor,
    decoder_input: torch.Tensor,
    teacher_forcing_ratio: float,
    criterion: nn.Module,
) -> torch.Tensor:
    last_output = None
    if_EOS_across_batch = torch.zeros(src_batch.size(0), device=device).bool()
    all_output_logits = PAD_token * torch.ones(
        (src_batch.size(0), MAX_SENTENCE_LENGTH, OUTPUT_TOKEN_SIZE), device=device
    )
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
        for t in range(0, MAX_SENTENCE_LENGTH):
            if last_output is not None and random.random() > teacher_forcing_ratio:
                # TODO: without this clone(), the embedding layer would expect a version lower than this
                # I still don't understand why. Have tried: using clone(), with torch.no_grad(), setting decoder_input.requires_grad = False
                # So let's keep it for now
                decoder_input = decoder_input.clone()
                decoder_input[:, t] = last_output.detach()
            # [batch size, t, output_vocab_dim]
            logits = model(
                src=src_batch,  # [batch size, max_length]
                tgt=decoder_input,  # [batch size, t]
            )
            last_logits = logits[:, t, :]  # [batch_size, output_vocab_dim]
            last_output = last_logits.argmax(
                -1
            )  # this won't change decoder_input's version
            all_output_logits[:, t, :] = last_logits  # length: max_sentence_length - 1

            # check if <EOS> appears across the entire batch
            if TERMINATE_TRAINING_UPON_EOS:
                if_EOS_across_batch |= last_output == EOS_token
                if if_EOS_across_batch.all():
                    break
        loss = criterion(
            all_output_logits.reshape(
                -1, logits.size(-1)
            ),  # (batch_size * sequence length, output_token_dim)
            tgt_batch.reshape(
                -1
            ),  # tgt_batch is ((batch_size-1) * sequence length). Need to take the first one out because we are not considering SOS now
        )
    if args.debug:
        training_logits_to_outuput_sentence(
            logits_batch=all_output_logits, tgt_batch=tgt_batch, output_lang=output_lang
        )
    return loss


def train_epoch(model, optimizer, criterion, dataloader, teacher_forcing_ratio):
    """Training Epoch with Mixed Precision and teacher forcing.

    Args:
        teacher_forcing_ratio (float): percentage of groundtruth being used.
            1.0 means using 100% groundtruth.

    Returns:
        Average loss
    """
    print(f"Teacher Forcing Ratio: {teacher_forcing_ratio}")
    total_loss = 0
    scaler = torch.amp.GradScaler(device=device, enabled=True)
    model.train()

    with torch.autograd.set_detect_anomaly(args.debug):
        with tqdm(total=len(dataloader), desc=f"Training", unit="batch") as pbar:
            for i, (src_batch, tgt_batch) in enumerate(dataloader):
                tgt_batch = tgt_batch.to(device)
                src_batch = src_batch.to(device)
                decoder_input = (
                    tgt_batch.clone().detach()
                )  # [batch_size, 1], a bunch of <SOS>
                decoder_input = decoder_input.to(device)
                batch_loss = train_with_teacher_enforcing(
                    src_batch=src_batch,
                    tgt_batch=tgt_batch,
                    decoder_input=decoder_input,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    criterion=criterion,
                )
                effective_batch_loss = batch_loss / ACCUMULATION_STEPS

                # calculate gradients
                scaler.scale(effective_batch_loss).backward()
                total_loss += effective_batch_loss.detach().item()
                # unscale gradients (from float16 to float32)
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    clip_gradients(model, GRADIENT_CLIPPED_NORM_MAX)
                    scaler.step(optimizer)
                    # Adjusts the scaling factor for the next iteration. If gradients are too low, increase the scaling factor.
                    scaler.update()
                    optimizer.zero_grad()

                pbar.update(1)

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, epochs, start_epoch):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Used for plotting later on
    wandb_logger = wandb.init(
        project="Torch-Transformer", resume="allow", anonymous="must"
    )
    wandb_logger.config.update(
        dict(
            epochs=NUM_EPOCHS,
            batch_size=EFFECTIVE_BATCH_SIZE,
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            max_sentence_length=MAX_SENTENCE_LENGTH,
            encoder_layer_num=ENCODER_DECODER_LAYER_NUM,
            decoder_layer_num=ENCODER_DECODER_LAYER_NUM,
            num_epochs=NUM_EPOCHS,
            teacher_forcing_ratio_min=TEACHER_FORCING_RATIO_MIN,
        )
    )
    timer = TrainingTimer()

    print("Training and validating model")
    scheduled_teacher_forcing_ratios = [
        get_scheduled_probability(1.0, TEACHER_FORCING_RATIO_MIN, epoch / epochs)
        for epoch in range(epochs)
    ]

    for epoch in range(start_epoch, epochs):
        torch.cuda.empty_cache()
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)
        train_loss = train_epoch(
            model,
            opt,
            loss_fn,
            train_dataloader,
            teacher_forcing_ratio=scheduled_teacher_forcing_ratios[epoch],
        )

        print(f"Training loss: {train_loss:.4f} \n")
        wandb_logger.log(
            {
                "epoch loss": train_loss,
                "epoch": epoch,
                "teacher forcing ratio": scheduled_teacher_forcing_ratios[epoch],
                "elapsed_time": timer.lapse_time(),
            }
        )
        save_model_and_optimizer(model, opt, epoch=epoch, path=MODEL_PATH)

    wandb.finish()


###############################################################################
# Validation Functions
###############################################################################


def tokens_to_words(tokens: torch.Tensor, lang: Lang) -> List[List[str]]:
    """Convert a batch of tokens to words

    Args:
        tokens (torch.Tensor): a batch of tokens

    Returns:
        [[words]]
    """
    # tokens: [batch_size, sentence_length]
    batch_translated_tokens = []
    for token_single_batch in tokens:
        translated_tokens = []
        for token in token_single_batch:
            token_idx = token.item()
            word = lang.index2word.get(token_idx, "<unk>")
            translated_tokens.append(word)
        batch_translated_tokens.append(translated_tokens)
    return batch_translated_tokens


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

        translated_tokens = tokens_to_words(tokens=ys, lang=output_lang)
        tgt_tokens = []
        try:
            tgt_tokens = tokens_to_words(tokens=tgt, lang=output_lang)
        except:
            # TODO: this is a hack for translate(). Please make this better when able.
            pass
        print(
            f"Translated sentence during training: {translated_tokens}, target: {tgt_tokens}"
        )


def src_batch_to_translated_tokens(model, src_batch: torch.Tensor) -> List[List[str]]:
    """Translate a batch of input tokens to a list of output tokens

    Args:
        src_batch (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    src_batch = src_batch.to(device)
    # TODO: this is a "semi-bug" - we are not getting the last token
    decoder_input = PAD_token * torch.ones(
        (src_batch.size(0), MAX_SENTENCE_LENGTH), dtype=torch.long, device=device
    )  # Shape: (1, 1)
    decoder_input[:, 0] = SOS_token
    decoder_input = decoder_input.to(device)
    for t in range(1, MAX_SENTENCE_LENGTH):
        # Only pass the portion of ys that we have generated so far
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            # Run the model with the truncated ys
            logits = model(
                src=src_batch,
                tgt=decoder_input,
            )
        last_logits = logits[:, t, :]  # [batch_size, output_vocab_dim]
        decoder_input[:, t] = last_logits.argmax(-1)
    translated_tokens = tokens_to_words(tokens=decoder_input, lang=output_lang)
    return translated_tokens


@torch.inference_mode()
def validate(model, dataloader):
    for i, (src_batch, tgt_batch) in enumerate(dataloader):
        translated_tokens = src_batch_to_translated_tokens(
            model=model, src_batch=src_batch
        )
        tgt_batch = tgt_batch.to(device)
        output_tokens = tokens_to_words(tokens=tgt_batch, lang=output_lang)
        target_tokens = [" ".join(o) for o in output_tokens]

        print(f" ==== \n")
        for tr, ta in zip(translated_tokens, target_tokens):
            print(f"Input: {ta}, \n translated_tokens: {tr} ")


@torch.inference_mode()
def translate(model, src_sentence, output_lang):
    src_tokens = input_lang_sentence_to_tokens(
        src_sentence=src_sentence, input_lang=input_lang
    )
    src_batch = PAD_token * torch.ones(
        (1, MAX_SENTENCE_LENGTH), dtype=torch.long, device=device
    )  # Shape: (1, 1)
    src_batch[0, : len(src_tokens)] = torch.tensor(src_tokens, dtype=torch.long)
    translated_tokens = src_batch_to_translated_tokens(model=model, src_batch=src_batch)
    print(f"Input: {src_sentence}, \n translated_tokens: {translated_tokens} ")


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # input_vocab_dim, embedding_dim, num_heads
    model = Transformer(
        input_token_size=INPUT_TOKEN_SIZE,
        output_token_size=OUTPUT_TOKEN_SIZE,
        dim_model=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=ENCODER_DECODER_LAYER_NUM,
        num_decoder_layers=ENCODER_DECODER_LAYER_NUM,
        dropout_p=0.1,
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_token)
    model, opt, start_epoch = load_model_and_optimizer(
        model,
        opt,
        path=MODEL_PATH,
        device=device,
    )
    if args.summary:
        input_size = [(MAX_SENTENCE_LENGTH,), (MAX_SENTENCE_LENGTH,)]
        model_summary(model, input_size, batch_dim=0, dtypes=[torch.long, torch.long])
    else:
        if not args.custom:
            if not args.eval:
                fit(
                    model,
                    opt,
                    loss_fn,
                    train_dataloader,
                    epochs=NUM_EPOCHS,
                    start_epoch=start_epoch,
                )
            model.eval()
            validate(model, train_dataloader)
        else:
            test_sentences = [
                "Eres tú",
                "Eres mala.",
                "Eres grande.",
                "Estás triste.",
                "estoy en el banco",
                "soy tom",
                "soy gorda",
                "estoy en forma",
                "Estoy trabajando.",
                "Estoy levantado.",
                "Estoy de acuerdo.",
            ]
            for test_sentence in test_sentences:
                translate(model, test_sentence, output_lang)
