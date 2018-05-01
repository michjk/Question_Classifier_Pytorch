import os
import shutil

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix

import sklearn

from tensorboard_logger import Logger

import dill as pickle

class PreprocessingPipelineWriter:
    def __init__(self, result_folder_path = "runs", saved_text_pipeline_file_path = "text_pipeline.pkl", saved_label_pipeline_file_path = "label_pipeline.pkl"):
        self.result_folder_path = result_folder_path
        self.saved_text_pipeline_file_path = os.path.join(result_folder_path, saved_text_pipeline_file_path)
        self.saved_label_pipeline_file_path = os.path.join(result_folder_path, saved_label_pipeline_file_path)
    
    def save_pipeline(self, data_pipeline, label=False):
        os.makedirs(self.result_folder_path, exist_ok=True)
        if not label:
            dirname = os.path.dirname(self.saved_text_pipeline_file_path)
            os.makedirs(dirname, exist_ok=True)
            pickle.dump(data_pipeline, open(self.saved_text_pipeline_file_path, 'wb'))
        else:
            dirname = os.path.dirname(self.saved_label_pipeline_file_path)
            os.makedirs(dirname, exist_ok=True)
            pickle.dump(data_pipeline, open(self.saved_label_pipeline_file_path, 'wb'))            
    
class PlotWriter:
    def __init__(self, path):
        self.plot_instance = Logger(path)
    
    def log_value(self, name, value, step):
        self.plot_instance.log_value(name, value, step)

class LearningWriter:
    def __init__(self, label_map, result_path = "runs", saved_model_file_path = "saved_model.model", train_log_folder_path = "train", dev_log_folder_path = "dev", confusion_matrix_folder_path = "confusion_matrix"):
        self.result_path = result_path
        self.saved_model_file_path = os.path.join(result_path, saved_model_file_path)
        self.train_log_folder_path = os.path.join(result_path, train_log_folder_path)
        self.dev_log_folder_path = os.path.join(result_path, dev_log_folder_path)
        self.confusion_matrix_file_path = os.path.join(result_path, confusion_matrix_folder_path, 'confusion_matrix.png')
        self.label_map = label_map

    def initialize(self):
        os.makedirs(self.result_path, exist_ok=True)
        
        self.train_logger = PlotWriter(self.train_log_folder_path)
        self.dev_logger = PlotWriter(self.dev_log_folder_path)
        
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


