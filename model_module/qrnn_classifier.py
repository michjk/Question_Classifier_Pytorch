import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torchqrnn.forget_mult
from torchqrnn import QRNN

class QRNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, pretrained_embedding_weight = None, train_embedding_layer = True, num_layers = 1, dropout = 0, zoneout = 0, window = 1, save_prev_x = False, use_gpu=True):
        '''
        Create QRNN classification module
        '''
        super().__init__()
        
        #initialize properties
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        #create nn module
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if not (pretrained_embedding_weight is None):
            self.word_embeddings.weight.data = pretrained_embedding_weight
            self.word_embeddings.weight.requires_grad = train_embedding_layer
        self.qrnn = QRNN(embedding_dim, hidden_dim, dropout=dropout, zoneout=zoneout, window = window, save_prev_x = save_prev_x, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_label = nn.Linear(hidden_dim, label_size)
        if use_gpu:
            self.cuda()
    
    def init_hidden(self):
        '''
        Initialize weight of hidden state
        '''
        self.hidden = autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        if self.use_gpu:
            self.hidden.cuda()
    
    def reset(self):
        self.qrnn.reset()
    
    def forward(self, sentence):
        self.batch_size = sentence.data.shape[1]
        self.init_hidden()
        
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        out, self.hidden = self.qrnn(x, self.hidden)
        out = self.dropout(out)
        y = self.hidden_to_label(out[-1])
        log_probs = F.log_softmax(y)
        return log_probs



