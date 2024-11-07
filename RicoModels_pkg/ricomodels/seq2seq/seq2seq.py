#!/usr/bin/env python3
from torch import nn, cuda
import torch
from torch import optim
import torch.nn.functional as F
from dataload_seq2seq import SOS_token, EOS_token, get_dataloader, tensorFromSentence
import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import random

from torch.utils.data import DataLoader  # Ensure DataLoader is imported if used

MAX_LENGTH = 30
MODEL_SAVE_PATH = "seq2seq_model.pth"
EPOCHS = 20

#######################################################
# Helper Functions
#######################################################

def to_device(data, device):
    """
    Recursively move tensors to the specified device.

    Args:
        data: A tensor or a collection of tensors (list, tuple, dict).
        device: The target device (e.g., torch.device('cuda')).

    Returns:
        The data moved to the specified device.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    # plt.show()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

#######################################################
# Model Definitions
#######################################################

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        hidden_dim = embed_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_batch):
        embedded = self.dropout(self.embedding(input_batch))
        #TODO Remember to remove
        print(f'input batch, embedded: {input_batch.shape, embedded.shape}')
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=self.device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.get_word_embedding(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def get_word_embedding(self, input, hidden):
        out = self.embedding(input)
        out = self.dropout(out)
        out = F.relu(out)
        out, hidden = self.gru(out, hidden)
        out = self.out(out)
        return out, hidden

#######################################################
# Training Infrastructure
#######################################################

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
               decoder_optimizer, criterion, device):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        # Move tensors to the specified device
        input_tensor, target_tensor = to_device(input_tensor, device), to_device(target_tensor, device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, device, learning_rate=0.001,
          print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer,
                          decoder_optimizer, criterion, device)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, input_lang, output_lang, device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_tensor = to_device(input_tensor, device).unsqueeze(0)  # Add batch dimension

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word.get(idx.item(), '<UNK>'))  # Handle unknown indices
    return decoded_words

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, device, n=100):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang, device)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def save_model(encoder, decoder, input_lang, output_lang, filepath):
    """
    Saves the state dictionaries of the encoder and decoder along with language vocabularies.

    Args:
        encoder (nn.Module): Trained encoder model.
        decoder (nn.Module): Trained decoder model.
        input_lang (Lang): Language object for input language.
        output_lang (Lang): Language object for output language.
        filepath (str): Path to save the model.
    """
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'input_lang': input_lang,
        'output_lang': output_lang
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, encoder, decoder, device):
    """
    Loads the state dictionaries of the encoder and decoder along with language vocabularies.

    Args:
        filepath (str): Path from where to load the model.
        device (torch.device): Device to map the loaded models.

    Returns:
        Tuple: (loaded_encoder, loaded_decoder, input_lang, output_lang)
    """
    checkpoint = torch.load(filepath, map_location=device)

    input_lang = checkpoint['input_lang']
    output_lang = checkpoint['output_lang']

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.to(device)
    decoder.to(device)

    print(f"Model loaded from {filepath}")
    return encoder, decoder, input_lang, output_lang

#######################################################
# Main Execution
#######################################################

if __name__ == "__main__":
    hidden_size = 128
    batch_size = 32

    # Define device
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load data
    input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size)

    # Initialize encoder and decoder with device
    encoder = Encoder(input_lang.n_words, hidden_size)
    decoder = Decoder(hidden_size, output_lang.n_words, device)

    # Optionally load a pre-trained model
    try:
        encoder, decoder, input_lang_loaded, output_lang_loaded = load_model(
            MODEL_SAVE_PATH,
            encoder,
            decoder,
            device
        )
    except FileNotFoundError:
        print("No pre-trained model found. Starting training from scratch.")

    encoder.to(device)
    decoder.to(device)

    # encoder.train()
    # decoder.train()

    # # Start training
    # train(train_dataloader, encoder, decoder, EPOCHS, device, print_every=5, plot_every=5)

    # # Save the trained models
    # save_model(encoder, decoder, input_lang, output_lang, MODEL_SAVE_PATH)

    # Switch to evaluation mode
    encoder.eval()
    decoder.eval()

    # Evaluate randomly
    evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, device)
