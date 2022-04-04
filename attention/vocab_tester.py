from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import contractions
import nltk
nltk.download('punkt')
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import json
import math


import time
import math


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1


MAX_LENGTH = 300


class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_to_vocab(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def sentence_to_indexes(self, sentence):
        indexes = [self.word2index[word] for word in sentence.split(' ')]
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def get_tensor_for_data(self, pair):
        input_tensor = self.sentence_to_indexes(pair[0])
        target_tensor = self.sentence_to_indexes(pair[1])
        return (input_tensor, target_tensor)


def preprocessString(text):
    text = text.lower().strip()
    text = contractions.fix(text)
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    return text

def read_data_and_vocab():
    print("Getting Data...")
    vocab = Vocabulary()
    
    data = []
    collated_df = pd.read_json('../../collated_full.json')
    collated_df = collated_df[collated_df["source"] != "The New York Times"]
    #collated_df = pd.read_json('training_data.json')
    print(f"Total Row Count : {len(collated_df.index)}")
    for index, row in collated_df.iterrows():

        text = row["text"]
        title = preprocessString(row["title"])

        sent_list = nltk.tokenize.sent_tokenize(text)
        num_of_sentences = len(sent_list)
        end_boundary = 3 if 3 < num_of_sentences else num_of_sentences
        first_3_sentences = sent_list[0:end_boundary]

        text = preprocessString(''.join(first_3_sentences))

        # print("setences: ",text)
        # print("ORI" ,row["text"])

        if(len(text.split(' ')) < MAX_LENGTH and len(title.split(' ')) < MAX_LENGTH):
            data.append([text,title])

            vocab.add_to_vocab(text)
            vocab.add_to_vocab(title)

    return data , vocab


data , VOCAB_MODEL = read_data_and_vocab()
print(f"Total Training Examples: {len(data)}")

t = VOCAB_MODEL.get_tensor_for_data(("have been some bomber shot along with","with along shot bomber some been have"))
print(t)
print(VOCAB_MODEL.index2word[int(t[0][0][0])])