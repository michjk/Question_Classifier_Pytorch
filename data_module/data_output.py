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

