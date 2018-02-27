import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
    Neural Network: CLSTM
    Detail: The input first cross CNN model ,then the output of CNN as the input of LSTM
"""


class CLSTM(nn.Module):
    
    def __init__(self, lstm_hidden_dim, lstm_num_layers, vocab_size, embedding_dim, label_size, kernel_num, kernel_sizes, dropout, batch_size, use_gpu = True):
        super(CLSTM, self).__init__()
        
        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.use_gpu = use_gpu
        
        V = vocab_size
        D = embedding_dim
        C = label_size
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        
        self.word_embeddings = nn.Embedding(V, D)
        
        # CNN
        KK = []
        for K in Ks:
            KK.append( K + 1 if K % 2 == 0 else K)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K//2, 0)) for K in KK])
        
        # LSTM
        self.lstm = nn.LSTM(Co, self.hidden_dim, num_layers= self.num_layers, dropout=dropout)
        self.hidden = self.init_hidden(batch_size)
        
        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)
        
        # dropout
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        if self.use_gpu is True:
            return (Variable(torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(1 * num_layers, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(1 * num_layers, batch_size, self.hidden_dim)))

    def forward(self, x):
        embed = self.word_embeddings(x)
        
        # CNN
        cnn_x = embed
        cnn_x = self.dropout(cnn_x)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 0)
        cnn_x = torch.transpose(cnn_x, 1, 2)
        
        # LSTM
        lstm_out, self.hidden = self.lstm(cnn_x, self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        
        # linear
        cnn_lstm_out = self.hidden2label1(F.tanh(lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))
        
        # output
        logit = F.log_softmax(cnn_lstm_out)

        return logit