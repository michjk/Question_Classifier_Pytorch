import numpy as np
import re
import itertools
from collections import Counter
import json

from tensorflow.contrib import learn

from torchtext import data
import codecs

label_place= set(['New_York_City', 'New_Haven,_Connecticut', 'Portugal', 'Southampton'])
label_event = set(['American_Idol', '2008_Sichuan_earthquake', '2008_Summer_Olympics_torch_relay', 'The_Blitz'])
label_person = set(['Beyonce', 'Frederic_Chopin', 'Queen_Victoria', 'Muammar_Gaddafi', 'Napoleon', 'Gamal_Abdel_Nasser', 'Dwight_D._Eisenhower', 'Kanye_West'])
label_period = set(['Buddhism', 'Hellenistic_period', 'Middle_Ages', 'Modern_history'])

place = "place"
event = "event"
person = "person"
period = "period"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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

def load_data_and_labels_highest_freq_clustered(test, num_of_top):
    #label_vectors = load_label_vector(label_file)
    
    questions = list(open(test, "r").readlines())
    x_text = [" ".join(s.split()[1:]) for s in questions]
    labels = [s.split()[0] for s in questions]
    x_text = [clean_str(s) for s in x_text]
    
    freq = Counter(labels)
    freq = freq.most_common(num_of_top)
    
    highest_labels = [label for label, _ in freq]

    set_highest_labels = set(highest_labels)

    y = []
    x = []

    print("Initial length ", len(x_text))
    len_questions = len(x_text)

    for i in range(len_questions):
        label = labels[i]
        text = x_text[i]

        if label in set_highest_labels:
            x.append(text)
            tmp = []
            if label in label_event:
                tmp = event
            if label in label_period:
                tmp = period
            if label in label_person:
                tmp = person
            if label in label_place:
                tmp = place
            
            y.append(tmp)
    
    print("Lenght after ", len(x), len(y))

    return x, y

def make_label_vector(list_labels):
    label_vectors = dict()
    vector_length = len(list_labels)

    i = 0
    for label in list_labels:
        init_vec = np.zeros(vector_length)
        init_vec[i] = 1
        label_vectors[label.strip()] = init_vec
        i += 1
    return label_vectors

def string_to_list_of_idx(x_text):
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    return x, vocab_processor

class FAQ(data.Dataset):
    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        # text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            questions, labels = load_data_and_labels_highest_freq_clustered(path, 20)
            len_questions = len(questions)

            for i in range(len_questions):
                examples.append(data.Example.fromlist([questions[i], labels[i]], fields))
            
            ''' with codecs.open(path,'r','utf8') as f:
                for line in f:
                    tmp = line.split()
                    x = " ".join(tmp[1:])
                    #x = clean_str(x)
                    y = tmp[0]
                    questions.append(x)
                    examples.append(data.Example.fromlist([x, y], fields)) '''
        
        super().__init__(examples, fields, **kwargs)
    
    @classmethod
    def splits(cls, text_field, label_field, path, dev_ratio, **kwargs):
        examples = cls(text_field, label_field, path, **kwargs).examples

        print(examples[279].text)
        print(examples[279].label)

        shuffle_indices = np.random.permutation(np.arange(len(examples)))
        np.random.shuffle(examples)

        dev_length = -1*int(dev_ratio*float(len(examples)))

        train_examples, dev_examples = examples[:dev_length], examples[dev_length:]

        print('train:',len(train_examples),'dev:',len(dev_examples))

        return (cls(text_field, label_field, examples=train_examples),
                cls(text_field, label_field, examples=dev_examples)
                )
    
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

def load_mr(text_field, label_field, batch_size, path, dev_ratio):
    print('loading data')
    train_data, dev_data = FAQ.splits(text_field, label_field, path, dev_ratio)

    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    
    print('building batches')
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data), batch_sizes=(batch_size, batch_size),
        repeat=False, device = None
    )
    
    return train_iter, dev_iter
