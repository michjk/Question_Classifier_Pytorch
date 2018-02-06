import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix

def createConfusionMatrix(truth_res, pred_res, path):
    confusion_matrix = ConfusionMatrix(truth_res, pred_res)
    confusion_matrix.plot()
    plt.savefig(path)
    print(confusion_matrix)
    confusion_matrix.print_stats()
