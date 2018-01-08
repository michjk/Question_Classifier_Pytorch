import time

import numpy as np

import torch
import torch.nn as nn

import torchqrnn.forget_mult
from torchqrnn import QRNN

class Model(nn.Module):
    def __init__(self, num_layers, hidden_size, vocab_size, batch_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = QRNN(hidden_size, hidden_size, num_layers)
    
    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.rnn(x)

if __name__ == 'main':
    main()

def main():
    print("hello")



