import time

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from model_module.lstm_classifier import LSTMClassifier

from data_module.data_preprocessor import *

import os
import random

import datetime

from model_module.model_runner import ModelRunner

from utils import load_parameters_from_json, filter_dotdict_class_propoperty

from data_module.data_writer import LearningLogger

import argparse

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path parameter json file")
param_json_path = parser.parse_args().path

parameter = load_parameters_from_json(param_json_path)
qrnn_parameter = filter_dotdict_class_propoperty(parameter, LSTMClassifier)

train_iter, dev_iter, vocab_size, label_size, pretrained_embedding_weight = load_dataset(
    parameter.dataset_path, parameter.batch_size, parameter.max_text_length, parameter.embedding_dim, dev_ratio = parameter.dev_ratio, pretrained_word_embedding_name = parameter.pretrained_word_embedding_name, pretrained_word_embedding_path = parameter.pretrained_word_embedding_path, use_gpu = parameter.use_gpu
    )

model = LSTMClassifier(**qrnn_parameter, vocab_size=vocab_size,label_size=label_size, pretrained_embedding_weight=pretrained_embedding_weight)

loss_function = nn.NLLLoss()
update_parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(update_parameter, lr = parameter.learning_rate, weight_decay=5e-4)
#optimizer = optim.Adagrad(update_parameter, lr=1e-3)
#optimizer = optim.RMSprop(update_parameter, lr=parameter.learning_rate, alpha=0.99, eps=1e-8, weight_decay=5e-4)

learning_logger = LearningLogger(parameter.result_folder_path, parameter.saved_model_file_path, parameter.train_log_folder_path, parameter.dev_log_folder_path)
mode_runner = ModelRunner(model, parameter.epoch, loss_function, optimizer, learning_logger,  parameter.transpose)
start_time = time.time()
mode_runner.learn(train_iter, dev_iter)
print("Overall time elapsed {} sec".format(time.time() - start_time))

