import torch
import torch.nn as nn
import torch.nn.functional as F

class  CNNClassifier(nn.Module):  
    def __init__(self, embedding_dim, vocab_size, label_size, batch_size, kernel_num, kernel_sizes, dropout = 0.5):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embedding_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, label_size)

    def forward(self, x):
        
        x = self.embed(x) # (N,W,D)
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        logloss = F.log_softmax(logit)

        return logloss