#!/usr/bin/env python3
import math

import torch
from torch import nn
from ricomodels.og_transformer.og_transformer import OGPositionalEncoder
from ricomodels.seq2seq.dataload_seq2seq import get_dataloader, EOS_token, SOS_token
import os
import argparse

BATCH_SIZE = 16
# NUM_KEYS = NUM_QUERIES = MAX_SENTENCE_LENGTH
NUM_KEYS = 50
NUM_QUERIES = 50
EMBEDDING_DIM = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
MAX_SENTENCE_LENGTH = 100
ENCODER_LAYER_NUM = 3
DECODER_LAYER_NUM = 3
NUM_EPOCHS = 30

input_lang, output_lang, train_dataloader, pairs = get_dataloader(BATCH_SIZE)
INPUT_TOKEN_SIZE = input_lang.n_words
OUTPUT_TOKEN_SIZE = output_lang.n_words
MODEL_PATH = "spanish_to_english.pth"

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = OGPositionalEncoder(MAX_SENTENCE_LENGTH, EMBEDDING_DIM)
    def forward( self, src, tgt, src_mask, tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer.encoder(
            src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask
        )
        outs = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.generator(outs)

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
    return mask

def create_mask(src, tgt):
    src_seq_len = src.size(0)
    tgt_seq_len = tgt.size(0)

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
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
        src_batch = src_batch.transpose(0, 1)
        tgt_batch = tgt_batch.transpose(0, 1)
        tgt_input = tgt_batch[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_batch, tgt_input)
        src_mask = src_mask.to(device=device) 
        tgt_mask = tgt_mask.to(device=device)
        src_padding_mask = src_padding_mask.to(device=device)  
        tgt_padding_mask = tgt_padding_mask.to(device=device) 
        logits = model(
            src_batch,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        optimizer.zero_grad()
        tgt_out = tgt_batch[1:, :]
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        src_mask = src_mask.to(device)
        memory = model.transformer.encoder(
            model.positional_encoding(model.src_tok_emb(src)),
            src_key_padding_mask=src_mask,
        )
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(0), device=device).type(torch.bool)
            # No need to call `.to(device)` again
            out = model.transformer.decoder(
                model.positional_encoding(model.tgt_tok_emb(ys)),
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_mask,
            )
            out = model.generator(out)
            prob = out[-1, :].softmax(dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.item()
            ys = torch.cat(
                [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
            )
            if next_word == EOS_token:
                break
    return ys

def translate(model, src_sentence):
    model.eval()
    src_tokens = [input_lang.word2index.get(word, 0) for word in src_sentence.split(' ')]
    src = torch.LongTensor(src_tokens).view(-1, 1).to(device)
    src_mask = (src == 0).transpose(0, 1).to(device)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=MAX_SENTENCE_LENGTH, start_symbol=SOS_token
    ).flatten()
    translated_tokens = [
        output_lang.index2word.get(token.item(), '<unk>') for token in tgt_tokens
    ]
    return ' '.join(translated_tokens[1:-1])  # Exclude SOS and EOS

def save_model(model, optimizer, epoch, path="transformer_model.pth"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"Model saved to {path}")

def load_model(model, optimizer, path="transformer_model.pth", device=torch.device('cpu')):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Model loaded from {path}, last trained epoch: {epoch}")
    
    # Move optimizer state to the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    return model, optimizer, epoch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", "-e", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = Seq2SeqTransformer(
        ENCODER_LAYER_NUM,
        DECODER_LAYER_NUM,
        EMBEDDING_DIM,
        NUM_HEADS,
        INPUT_TOKEN_SIZE,
        OUTPUT_TOKEN_SIZE,
        EMBEDDING_DIM,
    ).to(device)  # Move model to device immediately
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index
    
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # Load the model and optimizer state if checkpoint exists
    if os.path.exists(MODEL_PATH):
        model, optimizer, start_epoch = load_model(model, optimizer, path=MODEL_PATH, device=device)
    if not args.eval:
        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_loss = train_epoch(model, optimizer, train_dataloader)
            save_model(model, optimizer, epoch=NUM_EPOCHS, path=MODEL_PATH)
            print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')
    
    test_sentences = [
        "Hola.",
        "¡Corran!",
        "Corred."
        "¿Quién?"
    ]
    for test_sentence in test_sentences:
        translation = translate(model, test_sentence)
        print(f'Test Sentence: {test_sentence}, Translation: {translation}')
