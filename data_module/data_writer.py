import os
import shutil

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from tensorboard_logger import Logger

def createConfusionMatrix(truth_res, pred_res, path):
    confusion_matrix = ConfusionMatrix(truth_res, pred_res)
    confusion_matrix.plot()
    plt.savefig(path)
    print(confusion_matrix)
    confusion_matrix.print_stats()

class PlotLogger:
    def __init__(self, path):
        self.plot_instance = Logger(path)
    
    def log_value(self, name, value, step):
        self.plot_instance.log_value(name, value, step)

class LearningLogger:
    def __init__(self, result_path = "runs", saved_model_file_path = "saved_model.model", train_log_folder_path = "train", dev_log_folder_path = "dev"):
        self.result_path = result_path
        self.saved_model_file_path = os.path.join(result_path, saved_model_file_path)
        self.train_log_folder_path = os.path.join(result_path, train_log_folder_path)
        self.dev_log_folder_path = os.path.join(result_path, dev_log_folder_path)

    def initialize(self):
        print(self.result_path)
        if os.path.exists(self.result_path):
            shutil.rmtree(self.result_path)
            print("path erased")
        print(os.path.exists(self.result_path))
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        
        self.train_logger = PlotLogger(self.train_log_folder_path)
        self.dev_logger = PlotLogger(self.dev_log_folder_path)
        
        dirname = os.path.dirname(self.saved_model_file_path)
        os.makedirs(dirname, exist_ok=True)

    def train_log_value(self, name, value, step):
        self.train_logger.log_value(name, value, step)
    
    def dev_log_value(self, name, value, step):
        self.dev_logger.log_value(name, value, step)
    
    def save_model(self, model):
        torch.save(model, self.saved_model_file_path)
