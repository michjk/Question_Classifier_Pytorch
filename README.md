# Question_Classifier_Pytorch
Pytorch Implementation for question classifier

## Requirements
Please use Python 3, ubuntu 16.04, and CUDA 8. To install python library:
'''
pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
pip3 install cupy pynvrtc git+https://github.com/salesforce/pytorch-qrnn
pip3 install -r requirements.txt
'''
If failed please look requirements.txt and install one by one.
Git should be installed

## Dataset
The dataset should be seperated into training file and test file in CSV format
When preparing CSV file, the dataset should not use indexing (if saving using pandas, use pd.to_csv(file_name, index=False))

## Running training to save model and get test accuracy and loss

QRNN
'''
python train_qrnn.py --path train_rnn_parameter.json
'''

LSTM
'''
python train_lstm.py --path train_rnn_parameter.json
'''

CNN
'''
python train_cnn.py --path train_cnn_parameter.json
'''

## k-fold Cross Validation of model
QRNN
'''
python train_cv_qrnn.py --path train_rnn_parameter.json
'''

LSTM
'''
python train_cv_lstm.py --path train_rnn_parameter.json
'''

CNN
'''
python train_cv_cnn.py --path train_cnn_parameter.json
'''
Note: currently the validation result are printed to console 

## JSON file for training model
The JSON file are already prepared with appropriate setting
Most of the setting are for model hyperparameter
For now the important parameter that need to be change:
1. "train_dataset_path": train data path
2. "dev_dataset_path": test data path
3. "result_folder_path": where to save result such as best model, test accuracy (not available for cross validation)
4. "use_git": whether to use current commit information for better result versioning. if true result_folder_path become result_folder_path/branch_name_commit_date
5. n_folds: number of k-fold for cross validation
6. "transpose": set true for deploying CNN model, else false

Please look at the json file and try to run training first to understand more

## View test accuracy and loss with tensorboard
Please run
'''
tensorboard --logdir=[result_folder_path]
'''
To view graph of test loss and accuracy

##REST API server
To run in debug mode please run
'''
python rest_api_server.py --path rest_api_param.json
'''
The json file contain important parameter that need to be updated
1. "saved_model_file_path": path where the trained model saved
2. "text_vocab_path":  vocab information from questions in dataset (it should be in current directory after training)
3. "label_vocab_path": vocab information from labels in dataset (it should be in current directory after training)
4. "transpose": set true for deploying CNN model, else false

To run for production, we use uwsgi as server and nginx for receiving simultaneous
please look at https://hackernoon.com/a-guide-to-scaling-machine-learning-models-in-production-aa8831163846
The .ini file is already prepared as uwsgi.ini

