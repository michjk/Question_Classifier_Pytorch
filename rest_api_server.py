from flask import Flask
from flask import request

from model_module.cnn_classifier import CNNClassifier
from model_module.qrnn_classifier import QRNNClassifier
from model_module.lstm_classifier import LSTMClassifier

import torch
from torchtext import data

from data_module.data_preprocessor import QuestionWrapper, load_iter

import os
import re

from flask import jsonify

import logging

app = Flask(__name__)

MODEL_STATE_PATH = "faq_best_model.model"

DATASET_FOLDER = os.path.join("..", "dataset")
DATASET_PATH = os.path.join(DATASET_FOLDER, "faqs", "faq_ntu_prototype_v2.txt")

EMBEDDING_DIM = 300
EPOCH = 400
BATCH_SIZE = 64
DEV_RATIO = 0.1
DROPOUT = 0.5
KERNEL_SIZES = [3, 4, 5]
KERNEL_NUM = 128
MAX_TEXT_LENGHT = 81

def tokenizer(text): # create a tokenizer function
    text = text.lower()
    TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE) 
    return TOKENIZER_RE.findall(text)

text_field = data.Field(lower=True, tokenize=tokenizer, fix_length=MAX_TEXT_LENGHT)
label_field = data.Field(sequential=False)
train_iter, dev_iter = load_iter(text_field, label_field, batch_size=BATCH_SIZE, path = DATASET_PATH, dev_ratio=DEV_RATIO)

text_field.vocab.load_vectors('glove.6B.300d')

model = CNNClassifier(EMBEDDING_DIM, len(text_field.vocab), len(label_field.vocab)-1, BATCH_SIZE,KERNEL_NUM, KERNEL_SIZES, DROPOUT)
#model = model.cuda()
model.load_state_dict(torch.load(MODEL_STATE_PATH))

if (os.path.isfile("rest_api_logging.log")):
    os.remove("rest_api_logging.log")

# Create the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
 
# Create the Handler for logging data to a file
logger_handler = logging.FileHandler("rest_api_logging.log")
logger_handler.setLevel(logging.DEBUG)
 
# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
 
# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)
 
# Add the Handler to the Logger
logger.addHandler(logger_handler)
logger.info('Completed configuring logger()!')

def preprocess_question(question):
    question_data = QuestionWrapper(text_field, question)
    _, question_iter = data.Iterator.splits(
        (question_data, question_data), batch_size=len(question_data),
        repeat=False, device = -1
    )
    res = None

    for batch in question_iter:
        text = batch.text
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
        x = preprocess_question(question)
        model.eval()
        y = model(x)
        label_string = get_label(y)
        print(label_string)
        return jsonify({'result': str(label_string)})
    except:
        response = jsonify({'result': 'problem predicting'})
        response.status_code = 400
        return response

if __name__ == '__main__':
   app.run()