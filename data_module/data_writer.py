import os
import shutil

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pandas_ml import ConfusionMatrix
import seaborn as sns
import numpy as np
import sklearn

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
    def __init__(self, label_map, result_path = "runs", saved_model_file_path = "saved_model.model", train_log_folder_path = "train", dev_log_folder_path = "dev", confusion_matrix_folder_path = "confusion_matrix"):
        self.result_path = result_path
        self.saved_model_file_path = os.path.join(result_path, saved_model_file_path)
        self.train_log_folder_path = os.path.join(result_path, train_log_folder_path)
        self.dev_log_folder_path = os.path.join(result_path, dev_log_folder_path)
        self.confusion_matrix_file_path = os.path.join(result_path, confusion_matrix_folder_path, 'confusion_matrix.png')
        self.label_map = label_map

    def initialize(self):
        if os.path.exists(self.result_path):
            shutil.rmtree(self.result_path, ignore_errors=True)
            print("path erased")

        os.makedirs(self.result_path, exist_ok=True)
        
        self.train_logger = PlotLogger(self.train_log_folder_path)
        self.dev_logger = PlotLogger(self.dev_log_folder_path)
        
        dirname = os.path.dirname(self.saved_model_file_path)
        os.makedirs(dirname, exist_ok=True)

        dirname = os.path.dirname(self.confusion_matrix_file_path)
        os.makedirs(dirname, exist_ok=True)

    def train_log_value(self, name, value, step):
        self.train_logger.log_value(name, value, step)
    
    def dev_log_value(self, name, value, step):
        self.dev_logger.log_value(name, value, step)
    
    def save_model(self, model):
        torch.save(model, self.saved_model_file_path)
        print("model saved at", self.saved_model_file_path)

    def save_confusion_matrix(self, truth_res, pred_res):
        
        s = sklearn.metrics.confusion_matrix(truth_res, pred_res)
        df_cm = pd.DataFrame(data = s, columns=self.label_map, index=self.label_map)
        plt.figure(dpi=100)
        
        heatmap = sns.heatmap(df_cm, annot=True, fmt='d')
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=70, ha='right', fontsize=5)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=20, ha='right', fontsize=5)

        plt.savefig(self.confusion_matrix_file_path)

        confusion_matrix = ConfusionMatrix(truth_res, pred_res)
        confusion_matrix.print_stats()


