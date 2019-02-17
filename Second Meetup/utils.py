import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn import datasets


def create_mesh(X):
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    return grid, xx1, xx2

def plot_prediction(X_train, y_train, X_test, y_test, predict_func, params, fm_type='numpy'):
    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1)
    X_test = np.array(X_test)
    y_test = np.array(y_test).reshape(-1)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='b', label='Iris-setosa train')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='r', label='Others train')
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='b', label='Iris-setosa test', marker='+')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='r', label='Others test', marker='+')
    plt.legend()
    X = np.array(list(np.array(X_train)) + list(np.array(X_test)))
    grid, xx1, xx2 = create_mesh(X)
    if fm_type == 'tensorflow':
        probs = prediction.eval(feed_dict={X_data: grid.astype(np.float32)}).reshape(xx1.shape)
    elif fm_type == 'numpy':
        probs = predict_func(params['w'], params['b'], grid.T).reshape(xx1.shape)
    elif fm_type == 'keras':
        probs = predict_func(grid).reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
    
def accuracy(predictions, labels):
    return (100.0 * (np.sum((predictions.reshape(-1)>.5) == labels.reshape(-1)))
          / predictions.shape[0])