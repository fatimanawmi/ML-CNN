import numpy as np

from denseLayer import DenseLayer
from softmaxLayer import Softmax
from flattenLayer import Flatten
from reluActivaton import ReLU
from maxpool import MaxPool
from convLayer import ConvLayer

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import log_loss


class Network:
    def __init__(self, learning_rate):
        # lenet
        self.model = [ConvLayer(6, 5, 1, 0,learning_rate), ReLU(), MaxPool(2, 2), ConvLayer(16, 5, 1, 0,learning_rate), ReLU(), MaxPool(2, 2), Flatten(), DenseLayer(120,learning_rate), ReLU(), DenseLayer(84,learning_rate), ReLU(), DenseLayer(10,learning_rate), Softmax()]
    
    def forward(self, X):
        for layer in self.model:
            # print(X.shape, layer)
            X = layer.forward(X)
            if np.isnan(X).any() or np.isinf(X).any():
                print("X is nan or inf")
                print(X)
                print(layer)
                exit()
        return X

    def backward(self, y):
        grad = y
        for layer in reversed(self.model):
            # print if grad is nan or inf
            if np.isnan(grad).any() or np.isinf(grad).any():
                print("grad is nan or inf")
                print(grad)
                print(layer)
                exit()
            # print(grad.shape, layer)
            grad = layer.backward(grad)
        # return X

    def train(self, X, y):
        y_pred = self.forward(X)
        # print(X)
        self.backward(y)

    def cross_entropy_loss(self, y_pred, y_true):
        loss = log_loss(y_true, y_pred)
        return loss
    
    def f1_macro(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1
    
    def accuracy(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
    
    def confusion_matrix(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        confusion_matrix = multilabel_confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6,7,8,9])
        return confusion_matrix
    
    def clear(self):
        for layer in self.model:
            layer.clear()
