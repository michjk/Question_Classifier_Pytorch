import torch
import torch.nn as nn
import torch.nn.functional as F

class  CNNClassifier(nn.Module):

    '''
    CNNClassifier is Question Classifier base on CNN

    Args:
        embedding_dim (int): The size of word vector.
        vocab_size (int): The number of words/tokens in vocabulary.
        label_size (int): The number of possible labels.
        kernel_num (int): The number of kernels/filters for each kernels/filters size
        kernel_sizes (str/list): The list of kernel sizes. It can be in string such as "3, 4, 5" or in list such as [3, 4, 5]
        pretrained_embedding_weight (torch.Tensor): The pretrained word vectors (optional).
        train_embedding_layer (bool): Whether to train embedding layer or let embedding layer to be fixed.
        dropout (float): The probability of dropout's action.
        use_gpu (bool): Whether to use GPU or not
        layers: List of preconstructed QRNN layers to use for the QRNN module (optional).

    Inputs: x
        - x (seq_len, batch, input_size): tensor containing the features of the input sequence.
    Output: logsoftmax
        - logsoftmax (batch, label_size) : tensor result of log softmax
    '''
    def __init__(self, embedding_dim, vocab_size, label_size, kernel_num, kernel_sizes, pretrained_embedding_weight = None, train_embedding_layer = True, dropout = 0, use_gpu = True):
        super().__init__()
        
        #check if kernel sizes in str
        if isinstance(kernel_sizes, str):
            kernel_sizes = [int(i) for i in kernel_sizes.split(',')]

        #build network
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if not (pretrained_embedding_weight is None): #check if pretrain word vectors are used
            print("pretrain")
            self.word_embeddings.weight.data = pretrained_embedding_weight
            self.word_embeddings.weight.requires_grad = train_embedding_layer #need training or static
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embedding_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, label_size)

        #use gpu
        if use_gpu:
            self.cuda()
        
    def forward(self, x):
        x.t_() #transpose to match Conv2d input
        x = self.word_embeddings(x) # (N,W,D)
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        logloss = F.log_softmax(logit) # log of softmax
        return logloss