
# coding: utf-8

# In[1]:


import time

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from model_module.qrnn_classifier import QRNNClassifier

import torchwordemb

from data_module.data_preprocessor import *

import os
import random

import re
import datetime

import git

from model_module.model_runner import ModelRunner

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)

DATASET_FOLDER = os.path.join("..", "dataset")
DATASET_PATH = os.path.join(DATASET_FOLDER, "faqs", "faq_ntu_prototype_v6.txt") 

repo = git.Repo(os.getcwd())
headcommit = repo.head.commit
current_branch = repo.active_branch.name
RESULT_PATH = "runs/runs_" + current_branch + "_" + time.strftime("%a_%d_%b_%Y_%H_%M", time.gmtime(headcommit.committed_date))
SAVED_MODEL_FOLDER_PATH = os.path.join(RESULT_PATH, 'best_models')
SAVED_MODEL_NAME = 'faq_best_model.model'
LOG_FOLDER_PATH = os.path.join(RESULT_PATH, 'summaries')
TRAIN_LOG_FOLDER_PATH = 'train'
DEV_LOG_FOLDER_PATH = 'dev'

print(TRAIN_LOG_FOLDER_PATH)
print(DEV_LOG_FOLDER_PATH)

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
LAYERS_NUM = 1
EPOCH = 400
BATCH_SIZE = 32
DEV_RATIO = 0.1
DROPOUT = 0.5
ZONEOUT = 0.5
WINDOW = 2
SAVE_PREV_X = False
MAX_TEXT_LENGHT = None
USE_GPU = True
LEARNING_RATE = 1e-3

def tokenizer(text): # create a tokenizer function
    text = text.lower()
    TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE) 
    return TOKENIZER_RE.findall(text)

text_field = data.Field(lower=True, tokenize=tokenizer, fix_length=MAX_TEXT_LENGHT)
label_field = data.Field(sequential=False)
train_iter, dev_iter = load_iter(text_field, label_field, batch_size=BATCH_SIZE, path = DATASET_PATH, dev_ratio=DEV_RATIO)

model = QRNNClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1, batch_size=BATCH_SIZE, num_layers=LAYERS_NUM, dropout=DROPOUT, zoneout=ZONEOUT, window = WINDOW, save_prev_x = SAVE_PREV_X, use_gpu=USE_GPU)
''' 
vocab, vec = torchwordemb.load_word2vec_bin("../dataset/GoogleNews-vectors-negative300.bin")
text_field.vocab.set_vectors(vocab, vec, EMBEDDING_DIM)
'''
text_field.vocab.load_vectors('glove.6B.300d')

model.word_embeddings.weight.data = text_field.vocab.vectors.cuda()
model.word_embeddings.weight.requires_grad = False

loss_function = nn.NLLLoss()
update_parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(update_parameter, lr = LEARNING_RATE)
#optimizer = optim.Adagrad(update_parameter, lr=1e-3)
#optimizer = optim.RMSprop(update_parameter, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=5e-4)

mode_runner = ModelRunner(model, EPOCH, loss_function, optimizer, False, SAVED_MODEL_FOLDER_PATH, SAVED_MODEL_NAME, LOG_FOLDER_PATH, TRAIN_LOG_FOLDER_PATH, DEV_LOG_FOLDER_PATH)
start_time = time.time()
mode_runner.learn(train_iter, dev_iter)
print("Overall time elapsed {} sec".format(time.time() - start_time))

