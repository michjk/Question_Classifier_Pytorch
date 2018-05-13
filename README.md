# Question_Classifier_Pytorch
Pytorch Implementation for question classification model

## Requirements
The application is written in Python 3 and runs on Ubuntu 16.04 server with dedicated GPU. The following Ubuntu 16.04 packages should be installed:
1. Python 3
2. virtualenv
3. Git
4. CUDA 9/8
5. cuDNN 6
6. Nginx
7. uWSGI

Before running Python application or installing Python dependencies, please create a virtualenv and activate the respective virtualenv.
To create virtualenv:
```
virtualenv -p python3 virtualenv_name
```
To activate the created virtualenv:
```
source virtualenv_name/bin/activate
```
To install python depencies library for this appliaction:
```
pip install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl # for CUDA 9 or pip install torch for CUDA 8 
pip install cupy-cuda90 # or cupy-cuda80
pip install pynvrtc git+https://github.com/salesforce/pytorch-qrnn # for installing QRNN
pip install spacy
python -m spacy download en #for downloading English model for spaCy tokenizer
pip install cffi # probably not needed by just try
pip install -r requirements.txt
```
If it fails, please look at requirements.txt and install one by one.

## Dataset
The dataset should be seperated into training file and test file in CSV format.
When preparing CSV file, the dataset should not use indexing.
If you want to save CSV file using Pandas, use Python command:
```
pandas.to_csv('file_name', index=False)
```

## Project Structure
This project is basically my own framework for developing question classification. It is mostly built with Pytorch and also torchtext for Pytorch specific NLP preprocessor.
The framework suggest user to load module from 3 different folder.
1. data_module: The folder consist of function for loading & preprocessing dataset and writing results.
2. model_module: The folder consist of several classification models and a module for running training.

There is utils.py file which contains utility to load parameters from json file.
You can learn how to extend the framework from the current implementation.

## Models
Currently 3 different models are provided for question classification:
1. CNN (Convolutional Neural Network) 

Even though CNN is created for image related problem. It can be used for text classifiction. Some papers related to the CNN model are [
Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) and [
A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820).

2. LSTM (Long Short Term Memory)

LSTM is RNN based neural network architecture that solves vanishing gradient problem and long-term dependencies problem. It is a natural way to learn text pattern in sequential manner. The theory behind LSTM can be found in [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). With additional fully connected layer and softmax layer, LSTM can be used for text classification. 

3. QRNN (Quasi Recurrent Neural Networks) 

QRNN is RNN based neural network architecture that try to solve disadvantages of LSTM and CNN. LSTM is good for long-term dependency, but it is slow since it learn in sequential manner. Also, CNN can use parallelism with convolution function, but it cannot learn long-term dependencies. The theory behind QRNN is published in [Quasi Recurrent Neural Networks](https://arxiv.org/abs/1611.01576).

## Run training to save model and get test accuracy and loss
Example Python file are provided as example for training. For every training, please add new git commit if you indicate use_git = True in parameter json file to ensure there is no clash in saving result.

QRNN
```
python train_qrnn.py --path train_rnn_parameter.json
```

LSTM
```
python train_lstm.py --path train_rnn_parameter.json
```

CNN
```
python train_cnn.py --path train_cnn_parameter.json
```
The result are saved under result_folder_path. For accuracy & loss plot, use tensorboard to visualize them (will be explained).

## k-fold Cross Validation of model
QRNN
```
python train_cv_qrnn.py --path train_rnn_parameter.json
```

LSTM
```
python train_cv_lstm.py --path train_rnn_parameter.json
```

CNN
```
python train_cv_cnn.py --path train_cnn_parameter.json
```
Note: currently the cross validation result are printed to console, not saved 

## JSON file for training model
The JSON file are already prepared with appropriate setting. You can have parameter name different from example, but it is best to follow the example.Most of the setting are for model hyperparameter, loss function parameter, and optimizer parameter that has same parameter name as the model, loss function, and optimizer itselves.
Besides that, another parameter that important:
1. "epoch": number of epochs
2. "batch_size": batch size of training data
3. "max_text_length": max lenght of a sentence (must be indicated for CNN)
4. "train_dataset_path": train data path
5. "dev_dataset_path": test data path
6. "result_folder_path": where to save result such as best model, test accuracy (not available for cross validation)
7. "use_git": whether to use current commit information for better result versioning. if true result_folder_path become result_folder_path/branch_name_commit_date
8. "n_folds": number of k-fold for cross validation
9. "saved_model_file_path": file path relative to result_folder_path to save trained model
10. "saved_text_pipeline_file_path": file path relative to result_folder_path to save text preprocessing data
11. "saved_label_pipeline_file_path": file path relative to result_folder_path to save label preprocessing data
12. "train_log_folder_path":  file path relative to result_folder_path to save train plot for tensorboard
13. "dev_log_folder_path":  file path relative to result_folder_path to save test/evaluation plot for tensorboard.
14. "confusion_matrix_folder_path":  folder path relative to result_folder_path to save confusion matrix of test result.
15.  "pretrained_word_embedding_name": name of pretrained word embedding vectors. Currently support word2vec and GloVe (glove.6B.300d)
16. "pretrained_word_embedding_path": path to word embedding vectors file (support word2vec only example: "../dataset/GoogleNews-vectors-negative300.bin")

Please look at the json file and try to run training first to understand more

## View train & test accuracy and loss plot with tensorboard
Please run
```
tensorboard --logdir=[result_folder_path] --host 0.0.0.0
```
To view graph of test loss and accuracy.

## REST API server
To run in debug mode please run
```
python rest_api_server.py --path rest_api_param.json
```
The json file contain important parameter that need to be updated
1. "saved_model_file_path": path where the trained model saved
2. "debug_log_file_path": file path to debug log
3. "error_log_file_path": file path to error log
4. "saved_text_pipeline_file_path":  data.Field preprocessing information from questions in dataset (it should be in current directory after training)
5. "saved_label_pipeline_file_path": data.Field preprocessing information from labels in dataset (it should be in current directory after training)

To run for production, we use uwsgi as server and nginx for receiving simultaneous
please look at [A Guide to Scalling Machine Learning Models in Productions](https://hackernoon.com/a-guide-to-scaling-machine-learning-models-in-production-aa8831163846).
The .ini file is already prepared as uwsgi.ini

Currently, the endpoint to get prediction:
```
GET IP_ADDRESS/predict?question='question to be predicted'
```

## Telegram bot for trying model
There is available python file for running Telegram bot as UI for trying the classification model. Please look for tutorial how to make Telegram bot first. After that, run:
```
python telegram_bot.py --token TELEGRAM_TOKEN --ip http://0.0.0.0:port
```
--token is found after createing telegram bot. --ip is the target url to REST API model.
Also, google sheet can used for logging prediction and user suggestion, but it is not used by default. You can try to look how to setup google sheet api and use google_sheet_api.py as example.  