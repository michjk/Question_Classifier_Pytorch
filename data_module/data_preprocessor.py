import numpy as np
import re
import itertools
from collections import Counter
import json
from torchtext import data
import codecs
from sklearn.model_selection import KFold
import torchwordemb
import dill as pickle
import spacy

# load English model
spacy_nlp = spacy.load('en')

def clean_str(string):
    '''
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`@-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class QuestionWrapper(data.Dataset):
    '''
        A class wrapper for createing torchtext.data.Dataset instance for preprocessing question in production

        Inputs: text_field (data.Field), question (string)
            - text_field : loaded data.Field for preprocessing question
            - question : question sent by user
    '''
    def __init__(self, text_field, question, **kwargs):
        fields = [('text', text_field)]
        question = clean_str(question)
        examples = []
        examples.append(data.Example.fromlist([question], fields))

        super().__init__(examples, fields, **kwargs)

def preprocess_question(question, text_field, use_gpu = False):
    '''
        Preprocess question sent by user

        Inputs: question (string), text_field (data.Field), use_gpu (boolean)
            - question : question sent by user
            - text_field : loaded data.Field for preprocessing question
            - use_gpu : use gpu or not

        Outputs: text (Iterator)
            - text : Iterator of preprocessed question
    '''
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

        return text

def get_label(label_tensor, label_field):
    '''
        Get label string from tensor result

        Inputs: label_tensor (tensor), label_field (data.Field)
            - label_tensor: tensor result of classification
            - label_field: loaded data.Field for label preprocessing
    '''
    pred_index = label_tensor.data.max(1)[1]
    pred_index = pred_index.cpu().numpy()[0]
    label_string = label_field.vocab.itos[pred_index]

    return label_string

def tokenizer(text):
    '''
    Tokenizer, includes clear string and lemmatization

    Inputs:
        - text (string): string to be tokenized
    
    Outputs:
        - tokens (list): list of tokens

    '''

    #clean string
    text = clean_str(text)

    #lemmatized
    lemmatized = spacy_nlp(text)
    text = ' '.join([token.lemma_ for token in lemmatized])
    
    #tokenizer from tensorflow.preprocessing library
    tokenizer_re = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE) 
    return tokenizer_re.findall(text)

def load_dataset(train_path, dev_path, max_text_length, preprocessing_pipeline_writer, tokenizer = tokenizer, embedding_dim = 300, pretrained_word_embedding_name = "glove.6B.300d", pretrained_word_embedding_path = None):
    '''
        Load and preprocess dataset.

        Inputs:
            - train_path (string): Path to train dataset json
            - dev_path (string): Path to test/dev dataset json
            - max_text_length (integer): Max length of sentence. It pads dummy token up to max length
            - preprocessing_pipeline_writer (object): class for saving data.Field data
            - tokenizer (function): Tokenizer function
            - Embedding_dim (integer): size of word embedding dim
            - pretrained_word_embedding_name (string): name of pretrained word embedding. It can be glove.6B.300d or word2vec
            - pretrained_word_embedding_path (string): path to word embedding file. No need for glove.

        Outputs:
            - train_data (data.Dataset): train dataset
            - test_data (data.Dataset): test dataset
            - vocab_size (int): vocabulary size
            - label_size (int): size of possible label
            - label_vocab (dict): map from topic label string to index number
            - vectors : word embedding vectors
    '''

    # data.Field for preprocessing pipeline data
    text_field = data.Field(lower=True, tokenize=tokenizer, fix_length=max_text_length)
    label_field = data.LabelField()

    # load data
    print('loading data')
    train_data = data.TabularDataset(path=train_path, format='csv', skip_header=True, fields=[("text", text_field), ('label', label_field)])
    dev_data = data.TabularDataset(path=dev_path, format='csv', skip_header=True, fields=[("text", text_field), ('label', label_field)])
    
    # build vocabulary
    print('building vocab')
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)

    vectors = None

    # load word embedding vectors
    if pretrained_word_embedding_name == "word2vec":
        vocab, vec = torchwordemb.load_word2vec_bin(pretrained_word_embedding_path)
        text_field.vocab.set_vectors(vocab, vec, embedding_dim)
        vectors = text_field.vocab.vectors
    elif "glove" in pretrained_word_embedding_name:
        text_field.vocab.load_vectors(pretrained_word_embedding_name)
        vectors = text_field.vocab.vectors
    
    # save data.Field
    preprocessing_pipeline_writer.save_pipeline(text_field, False)
    preprocessing_pipeline_writer.save_pipeline(label_field, True)
    
    vocab_size = len(text_field.vocab)
    print("vocab size ", vocab_size)
    label_size = len(label_field.vocab)

    return train_data, dev_data, vocab_size, label_size, label_field.vocab.itos, vectors




    
