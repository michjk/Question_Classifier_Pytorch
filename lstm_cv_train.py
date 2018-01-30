
# coding: utf-8

# In[1]:


import time

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model_module.qrnn_classifier import QRNNClassifier
from model_module.lstm_classifier import LSTMClassifier

import torchwordemb

from data_module.data_preprocessor import *

import os
import random

import tensorboard_logger

import re
import datetime

import git

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)

DATASET_FOLDER = os.path.join("..", "dataset")
DATASET_PATH = os.path.join(DATASET_FOLDER, "faqs", "faq_ntu_prototype_v2.txt") 

repo = git.Repo(os.getcwd())
headcommit = repo.head.commit
current_branch = repo.active_branch.name
RESULT_PATH = "runs/runs_" + current_branch + "_" + time.strftime("%a_%d_%b_%Y_%H_%M", time.gmtime(headcommit.committed_date))

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
LAYERS_NUM = 1
EPOCH = 400
BATCH_SIZE = 64
DEV_RATIO = 0.1
DROPOUT = 0.5
ZONEOUT = 0.5
WINDOW = 2
SAVE_PREV_X = False
MAX_TEXT_LENGHT = None
n_splits = 10

def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
            right += 1.0
    return right/len(truth)

def evaluate(model, eval_iter, loss_function, i, name ='dev', eval_logger=None):
    if isinstance(model, QRNNClassifier):
        model.reset()
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
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc))
    if eval_logger:
        eval_logger.log_value("accuracy", acc, i)
        eval_logger.log_value("loss", avg_loss, i)
    return acc, avg_loss

def train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i, train_logger=None):
    if isinstance(model, QRNNClassifier):
        model.reset()
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
    if train_logger:
        train_logger.log_value("accuracy", acc, i)
        train_logger.log_value("loss", avg_loss, i)

def tokenizer(text): # create a tokenizer function
    text = text.lower()
    TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE) 
    return TOKENIZER_RE.findall(text)

text_field = data.Field(lower=True, tokenize=tokenizer, fix_length=MAX_TEXT_LENGHT)
label_field = data.Field(sequential=False)
train_dev_iter, test_iter = load_iter_cv(text_field, label_field, batch_size=BATCH_SIZE, path = DATASET_PATH, n_splits = n_splits, test_ratio=DEV_RATIO)

text_field.vocab.load_vectors('glove.6B.300d')

start_time = time.time()
cur_cv = 0
best_dev_acc_list = []
best_dev_loss_list = []
for train_iter, dev_iter in train_dev_iter:
    print("CV: ", cur_cv)
    best_dev_acc = 0.0
    best_dev_loss = 0.0

    model = model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1, batch_size=BATCH_SIZE, num_layers=LAYERS_NUM, dropout=DROPOUT)
    model = model.cuda()

    model.word_embeddings.weight.data = text_field.vocab.vectors.cuda()
    model.word_embeddings.weight.requires_grad = False

    loss_function = nn.NLLLoss()
    update_parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(update_parameter, lr = 1e-3)

    no_up = 0

    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i)
        print('now best dev acc:',best_dev_acc)
        dev_acc, dev_loss = evaluate(model,dev_iter, loss_function, i, 'dev')
        if dev_acc > best_dev_acc:
            no_up = 0
            best_dev_acc = dev_acc
            best_dev_loss = dev_loss
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                break
    
    cur_cv += 1
    best_dev_acc_list.append(best_dev_acc)
    best_dev_loss_list.append(best_dev_loss)

print("All cross validation accuracy: ", best_dev_acc_list)
print("Avg cross validation accuracy: ", np.average(best_dev_acc_list))

print("All cross validation loss: ", best_dev_loss_list)
print("Avg cross validation loss: ", np.average(best_dev_loss_list))

print("Overall time elapsed {} sec".format(time.time() - start_time))

