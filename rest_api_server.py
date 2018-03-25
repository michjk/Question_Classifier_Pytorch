from flask import Flask
from flask import request

import sys

import torch
from torchtext import data

import time

from data_module.data_preprocessor import get_label, preprocess_question

import os

from flask import jsonify

import logging

import dill as pickle

import argparse

from utils import *

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path parameter json file")
param_json_path = parser.parse_args().path

param = load_rest_api_parameter_from_json(param_json_path)

model = None
if param.use_gpu:
    model = torch.load(param.saved_model_file_path)
else:
    model = torch.load(param.saved_model_file_path, map_location=lambda storage, location: storage)

# Create the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
 
# Create the Handler for logging data to a file
logger_handler_debug = logging.FileHandler(param.debug_log_file_path)
logger_handler_debug.setLevel(logging.DEBUG)

# Create the Handler for logging data to a file
logger_handler_error = logging.FileHandler(param.error_log_file_path)
logger_handler_error.setLevel(logging.ERROR)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
 
# Add the Formatter to the Handler
logger_handler_debug.setFormatter(logger_formatter)
logger_handler_error.setFormatter(logger_formatter)
 
# Add the Handler to the Logger
logger.addHandler(logger_handler_debug)
logger.addHandler(logger_handler_error)
logger.info('Completed configuring logger()!')

text_field = pickle.load(open(param.text_vocab_path, "rb"))
label_field = pickle.load(open(param.label_vocab_path, "rb"))

@app.route('/predict', methods=['POST'])
def prediction():
    try:
        question = request.get_json()['question']
        logger.info("Question: " + question)
        x = preprocess_question(question, text_field, transpose = param.use_gpu, use_gpu=param.use_gpu)
        model.eval()
        t = time.time()
        y = model(x)
        dur = time.time() - t
        label_string = get_label(y, label_field)
        logger.info("Result: " + str(label_string))
        logger.info("Duration: " + str(dur))
        return jsonify({'result': str(label_string)})
    except:
        e = sys.exc_info()[0]
        logger.error("error ", str(e))
        response = jsonify({'error': str(e)})
        response.status_code = 400
        return response

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=8999)