
# coding: utf-8

# In[1]:


import time

import numpy as np

import torch
import torch.optim as optim

from data_module.data_preprocessor import *

import os
import random

from torchtext import data, datasets, vocab


# In[2]:


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


# In[3]:


EMBEDDING_DIM = 300
HIDDEN_DIM = 256
LAYERS_NUM = 1
EPOCH = 50
BATCH_SIZE = 24
DROPOUT = 0.3
ZONEOUT = 0


# In[4]:


text_field = data.Field()
label_field = data.Field(sequential=False)
train_data, test_data = datasets.IMDB.splits(text_field, label_field)


# In[5]:


text_field.build_vocab(train_data)
label_field.build_vocab(train_data)


# In[6]:


train_iter, test_iter = data.BucketIterator.splits((train_data, test_data), batch_size=BATCH_SIZE, device=None, repeat=False)


# In[7]:


text_field.vocab.load_vectors('glove.6B.300d')


# In[8]:


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


# In[9]:


from model_module.qrnn_classifier import QRNNClassifier

model = QRNNClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1, batch_size=BATCH_SIZE, num_layers=LAYERS_NUM, dropout=DROPOUT, zoneout=ZONEOUT)
model = model.cuda()


# In[10]:


model.word_embeddings.weight.data = text_field.vocab.vectors.cuda()


# In[13]:


from torch import nn

loss_function = nn.NLLLoss()
update_parameter = filter(lambda p: p.requires_grad, model.parameters())
#optimizer = optim.Adam(update_parameter, lr = 5e-4)
#optimizer = optim.Adagrad(update_parameter, lr=1e-3)
optimizer = optim.RMSprop(update_parameter, lr=1e-3, alpha=0.9, weight_decay=5e-4)


# In[15]:


from tensorboard_logger import configure, log_value
import datetime
mark = datetime.datetime.now()
configure("runs/runs_" + str(mark), flush_secs=2)


# In[16]:


start_time = time.time()
best_dev_acc = 0.0
no_up = 0
for i in range(EPOCH):
    print('epoch: %d start!' % i)
    train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i)
    print('now best dev acc:',best_dev_acc)
    dev_acc = evaluate(model,test_iter,loss_function,'dev')
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        os.system('rm best_models/mr_best_model_minibatch_acc_*.model')
        os.system('mkdir best_models')
        print('New Best Dev!!!')
        torch.save(model.state_dict(), 'best_models/mr_best_model_minibatch_acc_' + str(int(dev_acc*10000)) + '.model')
        no_up = 0
    
    ''' else:
        no_up += 1
        if no_up >= 10:
            exit() '''

print("Evaluate: {} sec".format(time.time()-start_time))