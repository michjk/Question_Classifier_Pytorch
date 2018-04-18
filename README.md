# Question_Classifier_Pytorch
Pytorch Implementation for question classification model

## Requirements
The application is written in Python 3 and runs on Ubuntu 16.04 server with dedicated GPU. The following Ubuntu 16.04 packages should be installed:
1. Python 3
2. virtualenv
3. Git
4. CUDA 8
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
pip install cupy //or cupy-cuda80 (for installing QRNN)
pip install pynvrtc git+https://github.com/salesforce/pytorch-qrnn //(for installing QRNN)
pip install -r requirements.txt
```
If it fails, please look requirements.txt and install one by one.

## Dataset
The dataset should be seperated into training file and test file in CSV format.
When preparing CSV file, the dataset should not use indexing.
If you want to save CSV file using Pandas, use Python command:
```
pandas.to_csv('file_name', index=False)
```

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
Example Python file are provided as example for training CNN
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
Note: currently the validation result are printed to console 

## JSON file for training model
The JSON file are already prepared with appropriate setting
Most of the setting are for model hyperparameter
For now the important parameter that need to be change:
1. train_dataset_path: train data path
2. dev_dataset_path: test data path
3. result_folder_path: where to save result such as best model, test accuracy (not available for cross validation)
4. use_git: whether to use current commit information for better result versioning. if true result_folder_path become result_folder_path/branch_name_commit_date
5. n_folds: number of k-fold for cross validation

Please look at the json file and try to run training first to understand more

## View test accuracy and loss with tensorboard
Please run
```
tensorboard --logdir=[result_folder_path]
```
To view graph of test loss and accuracy

## REST API server
To run in debug mode please run
```
python rest_api_server.py --path rest_api_param.json
```
The json file contain important parameter that need to be updated
1. saved_model_file_path: path where the trained model saved
2. text_vocab_path:  vocab information from questions in dataset (it should be in current directory after training)
3. label_vocab_path: vocab information from labels in dataset (it should be in current directory after training)
4. "transpose": set true for deploying CNN model, else false

To run for production, we use uwsgi as server and nginx for receiving simultaneous
please look at [A Guide to Scalling Machine Learning Models in Productions](https://hackernoon.com/a-guide-to-scaling-machine-learning-models-in-production-aa8831163846).
The .ini file is already prepared as uwsgi.ini

