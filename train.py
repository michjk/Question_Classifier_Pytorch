
# coding: utf-8

# In[1]:


import time

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchqrnn.forget_mult
from torchqrnn import QRNN

from data_module.data_preprocessor import *

import os
import random

from tensorboard_logger import configure, log_value

# In[2]:


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

DATASET_FOLDER = os.path.join("..", "dataset")
DATASET_PATH = os.path.join(DATASET_FOLDER, "faqs", "list_of_questions_train_labeled.txt")

EMBEDDING_DIM = 128
HIDDEN_DIM = 50
LAYERS_NUM = 3
EPOCH = 200
BATCH_SIZE = 64
DEV_RATIO = 0.1


# In[3]:


class QRNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, num_layers)
        self.qrnn = QRNN(embedding_dim, hidden_dim)
        self.hidden_to_label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda())
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        out, self.hidden = self.qrnn(x, self.hidden)
        y = self.hidden_to_label(out[-1])
        log_probs = F.log_softmax(y)
        return log_probs
        


# In[4]:


def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
            right += 1.0
    return right/len(truth)


# In[5]:


def evaluate(model, eval_iter, loss_function,  name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    print(eval_iter)
    for batch in eval_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred = model(sent)
        pred_label = pred.data.max(1)[1]
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]

    avg_loss /= len(eval_iter)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
    log_value('Accuracy', acc, i)
    log_value('Loss', avg_loss, i)
    return acc


# In[6]:


def train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    for batch in train_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()# detaching it from its history on the last instance.
        pred = model(sent)
        pred_label = pred.data.max(1)[1]
        #print(pred_label)
        #print(len(pred_label))
        #print(label)
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        if count % 100 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.data[0]))
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res,pred_res)
    print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, acc))


# In[7]:


import re

def tokenizer(text): # create a tokenizer function
    text = text.lower()
    TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
    return TOKENIZER_RE.findall(text)

text_field = data.Field(lower=True, tokenize=tokenizer)
label_field = data.Field(sequential=False)
train_iter, dev_iter = load_mr(text_field, label_field, batch_size=BATCH_SIZE, path = DATASET_PATH, dev_ratio=DEV_RATIO)


# In[8]:


best_dev_acc = 0.0

model = QRNNClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1, batch_size=BATCH_SIZE, num_layers=LAYERS_NUM)
model = model.cuda()


# In[9]:


loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)


# In[10]:

configure("runs/run_1", flush_secs=2)

no_up = 0
for i in range(EPOCH):
    print('epoch: %d start!' % i)
    train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i)
    print('now best dev acc:',best_dev_acc)
    dev_acc = evaluate(model,dev_iter,loss_function,'dev')
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        os.system('rm best_models/mr_best_model_minibatch_acc_*.model')
        os.system('mkdir best_models')
        print('New Best Dev!!!')
        torch.save(model.state_dict(), 'best_models/mr_best_model_minibatch_acc_' + str(int(dev_acc*10000)) + '.model')
        no_up = 0
    ''' 
    else:
        no_up += 1
        if no_up >= 10:
            exit()
    '''

