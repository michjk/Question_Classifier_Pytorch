import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torchqrnn.forget_mult
from torchqrnn import QRNN

class QRNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, num_layers = 1, dropout = 0, zoneout = 0, window = 1, save_prev_x = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.qrnn = QRNN(embedding_dim, hidden_dim, dropout=dropout, zoneout=zoneout, window = window, save_prev_x = save_prev_x, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        """
        Initialize weight of hidden
        """
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda())
    
    def reset(self):
        self.qrnn.reset()
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        out, self.hidden = self.qrnn(x, self.hidden)
        out = self.dropout(out)
        y = self.hidden_to_label(out[-1])
        log_probs = F.log_softmax(y)
        return log_probs


