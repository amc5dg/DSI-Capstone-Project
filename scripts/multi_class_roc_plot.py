import matplotlib.pyplot as plt
import numpy as np
from clean_test import load_clean_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from predict import *


classes = ['Edge-on Disk', 'Elliptical', 'Face-on Spiral', 'Merger']
path_to_project_data = '/home/ubuntu/DSI-Capstone-Project/data/'


def get_rates(y_true, y_pred, thresh=1):
    '''
    input: y_true (np array), y_pred (np array)
    output: TPR (float), FPR (float)
    '''
    # set y-pred by threshold ; calculate TP and FP
    y_pred = y_pred >= thresh
    TP = ((y_pred == y_true) & (y_true == 1)).sum()
    FP = ((y_pred != y_true) & (y_true == 0)).sum()
    # calculate TPR AND FPR
    TPR = float(TP) / (y_true == 1).sum()
    FPR = float(FP) / (y_true == 0).sum()
    return TPR, FPR


def plot_one_curve(y_test, y_pred, label):
    tprs, fprs = [], []
    for thresh in sorted(y_pred):
        TPR, FPR = get_rates(y_test, y_pred, thresh=thresh)
        tprs.append(TPR)
        fprs.append(FPR)
	# plots
    plt.plot(fprs, tprs, label = label)



def plot_roc(y_test, probs, plotname = 'roc_plt.png'):
	'''
	input: y_test (np array), probs (np array), plotname (string)
	output: None (saves plot to disk)
	'''
    plt.clf()
    # plots 45 degree angle line for reference
    z = np.linspace(0,1)
    plt.plot(z, z, ls='dotted')
	for i, lbl in enumerate(classes):
		plot_one_curve(y_test, probs[:,i], lbl)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
	plt.legend()
    plt.savefig(plotname)


if __name__ == '__main__':
    model = load_model('CNN_model_architecture.json', 'model_weights.h5')
	X_test = np.load(path_to_project_data+'X_test_all.npy')
    y_test = np.load(path_to_project_data+'y_test_all.npy')

    # to predict on many images
    probs = predict_many(model, X_test)

    plot_roc(y_test, probs)
