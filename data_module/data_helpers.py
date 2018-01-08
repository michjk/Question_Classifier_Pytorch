import numpy as np
import re
import itertools
from collections import Counter
import json

label_place= set(['New_York_City', 'New_Haven,_Connecticut', 'Portugal', 'Southampton'])
label_event = set(['American_Idol', '2008_Sichuan_earthquake', '2008_Summer_Olympics_torch_relay', 'The_Blitz'])
label_person = set(['Beyonce', 'Frederic_Chopin', 'Queen_Victoria', 'Muammar_Gaddafi', 'Napoleon', 'Gamal_Abdel_Nasser', 'Dwight_D._Eisenhower', 'Kanye_West'])
label_period = set(['Buddhism', 'Hellenistic_period', 'Middle_Ages', 'Modern_history'])

label_dimmension = label_place | label_event | label_period

place = "place"
event = "event"
person = "person"
period = "period"

dimmension = "dimmension"

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
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_data_and_labels_FAQs(train):
    #examples = list(open(train, "r", encoding="utf-8").readlines())
    examples = list(open(train, "r").readlines())
    x_text = [" ".join(s.split()[1:]) for s in examples]
    x_text = [clean_str(sent) for sent in x_text]
    y = []
    for s in examples:
        if s.startswith("PROG"):
            y.append([1, 0, 0, 0, 0])
            continue
        if s.startswith("SCHS"):
            y.append([0, 1, 0, 0, 0])
            continue
        if s.startswith("ACCO"):
            y.append([0, 0, 1, 0, 0])
            continue
        if s.startswith("ADMI"):
            y.append([0, 0, 0, 1, 0])
            continue
        if s.startswith("SCHO"):
            y.append([0, 0, 0, 0, 1])
            continue
        #if s.startswith("CONT"):
        #    y.append([0, 0, 0, 0, 0, 1])
        #    #y.append([0, 1])
        #    continue
        else:
            print("Error: Category is not defined", s)
            exit()
    y = np.concatenate([y], 0)
    return [x_text, y]

def load_data_and_labels_eval_FAQs(test, label_file):
    label_vectors = load_label_vector(label_file)
    #print (label_vectors.keys())
    
    examples = list(open(test, "r").readlines())
    x_text = [" ".join(s.split()[1:]) for s in examples]
    #x_text = [clean_str(sent) for sent in x_text]
    y = []
    for s in examples:
        label = s.split()[0]
        y.append(label_vectors[label])
    y = np.concatenate([y], 0)
    return [x_text, y]

def load_data_and_labels_highest_freq(test, num_of_top):
    #label_vectors = load_label_vector(label_file)
    
    questions = list(open(test, "r").readlines())
    x_text = [" ".join(s.split()[1:]) for s in questions]
    labels = [s.split()[0] for s in questions]
    x_text = [clean_str(s) for s in x_text]
    
    freq = Counter(labels)
    freq = freq.most_common(num_of_top)
    
    highest_labels = [label for label, _ in freq]
    set_highest_labels = set(highest_labels)
    
    label_vectors = make_label_vector(highest_labels)

    y = []
    x = []

    print("Initial length ", len(x_text))
    len_questions = len(x_text)
    
    for i in range(len_questions):
        label = labels[i]
        text = x_text[i]

        if label in set_highest_labels:
            x.append(text)
            y.append(label_vectors[label])
    
    print("Lenght after ", len(x), len(y))

    y = np.concatenate([y], 0)

    print("After concatenate ", len(y))
    return [x, y]

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
    
    label_vectors = make_label_vector([event, period, person, place])

    y = []
    x = []

    print("Initial length ", len(x_text))
    len_questions = len(x_text)
    
    count_new_label = []

    for i in range(len_questions):
        label = labels[i]
        text = x_text[i]

        if label in set_highest_labels:
            x.append(text)
            tmp = []
            if label in label_event:
                tmp = label_vectors[event]
                count_new_label.append(event)
            if label in label_period:
                tmp = label_vectors[period]
                count_new_label.append(period)
            if label in label_person:
                tmp = label_vectors[person]
                count_new_label.append(person)
            if label in label_place:
                tmp = label_vectors[place]
                count_new_label.append(place)
            
            y.append(tmp)
    
    print("Lenght after ", len(x), len(y))

    count_new_label = Counter(count_new_label)

    print(count_new_label)

    y = np.concatenate([y], 0)

    print("After concatenate ", len(y))
    return [x, y]

def load_data_and_labels_highest_freq_clustered_2(test, num_of_top):
    #label_vectors = load_label_vector(label_file)
    
    questions = list(open(test, "r").readlines())
    x_text = [" ".join(s.split()[1:]) for s in questions]
    labels = [s.split()[0] for s in questions]
    x_text = [clean_str(s) for s in x_text]
    
    freq = Counter(labels)
    freq = freq.most_common(num_of_top)
    
    highest_labels = [label for label, _ in freq]

    set_highest_labels = set(highest_labels)
    
    label_vectors = make_label_vector([dimmension, person])

    y = []
    x = []

    print("Initial length ", len(x_text))
    len_questions = len(x_text)
    
    count_new_label = []

    for i in range(len_questions):
        label = labels[i]
        text = x_text[i]

        if label in set_highest_labels:
            x.append(text)
            tmp = []

            if label in label_person:
                tmp = label_vectors[person]
                count_new_label.append(person)
            if label in label_dimmension:
                tmp = label_vectors[dimmension]
                count_new_label.append(dimmension)
            
            y.append(tmp)
    
    print("Lenght after ", len(x), len(y))

    count_new_label = Counter(count_new_label)

    print(count_new_label)

    y = np.concatenate([y], 0)

    print("After concatenate ", len(y))
    return [x, y]

def load_data_and_labels_clustered_2(test):
    questions = list(open(test, "r").readlines())
    x_text = [" ".join(s.split()[1:]) for s in questions]
    labels = [s.split()[0] for s in questions]
    x_text = [clean_str(s) for s in x_text]
    
    label_vectors = make_label_vector([dimmension, person])

    y = []
    x = []

    print("Initial length ", len(x_text))
    len_questions = len(x_text)

    for i in range(len_questions):
        label = labels[i]
        text = x_text[i]
        x.append(text)
        if i%2 == 1:
            y.append(label_vectors[dimmension])
        else:
            y.append(label_vectors[person])
    
    print("Lenght after ", len(x), len(y))

    y = np.concatenate([y], 0)

    print("After concatenate ", len(y))
    return [x, y]

def load_class_vector(class_str):
    y = []
    s = class_str
    if s.startswith("PROG"):
        y.append([1, 0, 0, 0, 0, 0])
    if s.startswith("SCHS"):
        y.append([0, 1, 0, 0, 0, 0])
    if s.startswith("ACCO"):
        y.append([0, 0, 1, 0, 0, 0])
    if s.startswith("ADMI"):
        y.append([0, 0, 0, 1, 0, 0])
    if s.startswith("SCHO"):
        y.append([0, 0, 0, 0, 1, 0])
    if s.startswith("CONT"):
        y.append([0, 0, 0, 0, 0, 1])
    return y

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

def load_label_vector(label_file):
    examples = list(open(label_file, "r").readlines())
    label_vectors = dict()
    vector_length = len(examples)
    
    i = 0
    for label in examples:
        init_vec = np.zeros(vector_length)
        init_vec[i] = 1
        label_vectors[label.strip()] = init_vec
        i = i + 1
    return label_vectors

def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors