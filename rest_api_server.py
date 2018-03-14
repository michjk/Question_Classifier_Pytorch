from flask import Flask
from flask import request

import torch
from torchtext import data

from data_module.data_preprocessor import QuestionWrapper

import os
import re

from flask import jsonify

import logging

import dill as pickle

import argparse

from utils import *

app = Flask(__name__)

MODEL_STATE_PATH = "runs/refactor_Tue_13_Mar_2018_17_58/best_model/faq_best_model.model"

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path parameter json file")
param_json_path = parser.parse_args().path

param = load_parameters_from_json(param_json_path)

model = None
if param.use_gpu:
    model = torch.load(param.saved_model_file_path)
else:
    model = torch.load(param.saved_model_file_path, map_location=lambda storage, location: storage)
""" 
if (os.path.isfile(path.saved_model_file_path)):
    os.remove("rest_api_logging.log")
"""
# Create the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
 
# Create the Handler for logging data to a file
logger_handler = logging.FileHandler(param.log_file_path)
logger_handler.setLevel(logging.DEBUG)
 
# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
 
# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)
 
# Add the Handler to the Logger
logger.addHandler(logger_handler)
logger.info('Completed configuring logger()!')

text_field = pickle.load(open(param.text_vocab_path, "rb"))
label_field = pickle.load(open(param.label_vocab_path, "rb"))

def preprocess_question(question, transpose = False, use_gpu = False):
    device = -1
    if use_gpu:
        device = None
    question_data = QuestionWrapper(text_field, question)
    _, question_iter = data.Iterator.splits(
        (question_data, question_data), batch_size=len(question_data),
        repeat=False, device = device
    )

    for batch in question_iter:
        text = batch.text
        if transpose:
            text.data.t_()

        return text

def get_label(label_tensor):
    pred_index = label_tensor.data.max(1)[1]
    pred_index = pred_index.cpu().numpy()[0]
    label_string = label_field.vocab.itos[pred_index+1]

    logger.info(label_string)

    return label_string

@app.route('/predict', methods=['POST'])
def prediction():
    try:
        print("Predicting")
        print(request.method)
        question = request.get_json()['question']
        print(question)
        logger.info(question)
        x = preprocess_question(question, transpose = param.use_gpu, use_gpu=param.use_gpu)
        print("preprocess complete")
        model.eval()
        y = model(x)
        print("model inference")
        label_string = get_label(y)
        print(label_string)
        return jsonify({'result': str(label_string)})
    except:
        response = jsonify({'result': 'problem predicting'})
        response.status_code = 400
        return response

if __name__ == '__main__':
   app.run(host=param.host)