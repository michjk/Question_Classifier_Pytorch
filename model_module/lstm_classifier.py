import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torchqrnn.forget_mult
from torchqrnn import QRNN

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        """
        Initialize weight of hidden
        """
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda())
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        out, self.hidden = self.lstm(x, self.hidden)
        out = self.dropout(out)
        y = self.hidden_to_label(out[-1])
        log_probs = F.log_softmax(y)
        return log_probs


