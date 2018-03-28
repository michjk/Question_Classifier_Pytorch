import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torchqrnn.forget_mult
from torchqrnn import QRNN

class QRNNClassifier(nn.Module):
    '''
    QRNNClassifier is classification module based on QRNN
    Args:
        embedding_dim (int): The size of word vector.
        vocab_size (int): The number of words/tokens in vocabulary.
        label_size (int): The number of possible labels.
        hidden_dim: The number of features in the hidden state h of QRNN
        pretrained_embedding_weight (torch.Tensor): The pretrained word vectors (optional).
        train_embedding_layer (bool): Whether to train embedding layer or let embedding layer to be fixed.
        num_layers (int): The number of layers of QRNN.
        save_prev_x (bool): Whether to store previous inputs for use in future convolutional windows of QRNN (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x (reset is not yet implemented). Default: False.
        window (int): Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        dropout (float): The probability of dropout in QRNN and dropout layer.
        zoneout (float): Whether to apply zoneout of QRNN (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        use_gpu (bool): Whether to use GPU or not
    
    Inputs: x:
        - x (seq_len, batch, input_size): tensor containing the features of the input sequence.
    
    Output: logsoftmax
        - logsoftmax (batch, label_size) : tensor result of log softmax
    '''
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, pretrained_embedding_weight = None, train_embedding_layer = True, num_layers = 1, dropout = 0, zoneout = 0, window = 1, save_prev_x = False, use_gpu=True):
        super().__init__()
        
        #initialize properties
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.use_pretrained_word_embedding = not (pretrained_embedding_weight is None)
        self.train_embedding_layer = train_embedding_layer
        self.pretrained_embedding_weight = pretrained_embedding_weight

        #create nn module
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if self.use_pretrained_word_embedding:
            self.word_embeddings.weight.data = pretrained_embedding_weight
            self.word_embeddings.weight.requires_grad = train_embedding_layer
        self.qrnn = QRNN(embedding_dim, hidden_dim, dropout=dropout, zoneout=zoneout, window = window, save_prev_x = save_prev_x, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_label = nn.Linear(hidden_dim, label_size)
        
        #use gpu or not
        if use_gpu:
            self.cuda()
    
    def init_hidden(self):
        '''
        Initialize weight of hidden state
        '''
        self.hidden = autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        if self.use_gpu:
            self.hidden.cuda()
    
    def forward(self, sentence):
        #get batch size and init hidden
        self.batch_size = sentence.data.shape[1]
        self.init_hidden()
        
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        out, self.hidden = self.qrnn(x, self.hidden)
        out = self.dropout(out)
        y = self.hidden_to_label(out[-1])
        log_probs = F.log_softmax(y)
        
        return log_probs



