#!/usr/bin/env python3

from __future__ import division, print_function, unicode_literals

import random
import re
import unicodedata
from io import open

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from ricomodels.utils.data_loading import get_package_dir
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 1
EOS_token = 2
PAD_token = 0
MAX_LENGTH = 30


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    # return a string
    return s.strip()


def readLangs(lang1, lang2, reverse=False):
    # use this to preprocess spa.txt
    # sed 's/CC-BY 2\.0.*//' spa.txt > spa-eng.txt
    print("Reading lines...")

    # Read the file and split into lines
    package_dir = get_package_dir()
    data_file_path = os.path.join(package_dir, f"seq2seq/{lang1}-{lang2}.txt")
    lines = open(data_file_path, encoding="utf-8").read().strip().split("\n")

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split("\t") if s] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s ",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re ",
)


def filterPair(p):
    # MAX_LENGTH-2 because we are considering <SOS> and <EOS>
    return (
        len(p[0].split(" ")) <= MAX_LENGTH - 2
        and len(p[1].split(" ")) <= MAX_LENGTH - 2
        # TODO: experimenting
        # and p[1].startswith(eng_prefixes)
    )


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


class Lang:
    """
    Build a counter, and word <-> index lookups for every new word.
    This will help set up one-hot vectors. Key components are:
    - word2index: each word has an index.
    - word2count: counting words.
    """

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS", PAD_token: "PAD"}
        self.n_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def prepareData(lang1, lang2, reverse=False):
    """Return input and output sentences in lists, also, update tokens based on the number of words"""
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return (
        [SOS_token]
        + [lang.word2index[word] for word in sentence.split(" ")]
        + [EOS_token]
    )


def get_dataloader(batch_size):
    """Generate sequences with tokens

    Each sequence is [<SOS>, <token1>...<EOS>, <PAD>, <PAD>... ]
    (MAX_SENTENCE_LENGTH)

    """

    # 🪦 Reverse=True is weird, I feel ya. It's because the dataset we use spa-eng.txt actually has english on the left
    # Spanish on the right. TODO: input language here is "eng", but really it's spanish content. It doesn't break my code now,
    # but needs to be fixed
    input_lang, output_lang, pairs = prepareData("spa", "eng", reverse=True)

    n = len(pairs)
    input_ids = PAD_token * np.ones((n, MAX_LENGTH), dtype=np.int32)
    target_ids = PAD_token * np.ones((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids

    for i in range(10):
        print(f"input:{pairs[i]}")
    print(f"Rico: example token: {input_ids[0]}")
    print(f"Number of sentences is {len(pairs)}")

    train_data = TensorDataset(
        torch.LongTensor(input_ids), torch.LongTensor(target_ids)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    return input_lang, output_lang, train_dataloader, pairs


def input_lang_sentence_to_tokens(src_sentence, input_lang):
    src_tokens = (
        [SOS_token]
        + [
            input_lang.word2index[normalizeString(word)]
            for word in src_sentence.split(" ")
        ]
        + [EOS_token]
    )

    return src_tokens


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData("spa", "eng", True)
    print("input_lang, output_lang, pairs:", f"{input_lang, output_lang, pairs}")
