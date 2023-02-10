import numpy as np

from denseLayer import DenseLayer
from softmaxLayer import Softmax
from flattenLayer import Flatten
from reluActivaton import ReLU
from maxpool import MaxPool
from convLayer import ConvLayer


class Network:
    def __init__(self):
        # lenet
        self.model = [ConvLayer(6, 5, 1, 0), ReLU(), MaxPool(2, 2), ConvLayer(16, 5, 1, 0), ReLU(), MaxPool(2, 2), Flatten(), DenseLayer(120), ReLU(), DenseLayer(84), ReLU(), DenseLayer(10), Softmax()]
        #alexnet
        # self.model = [ConvLayer(96, 11, 4, 0), ReLU(), MaxPool(2, 2), ConvLayer(256, 5, 1, 2), ReLU(), ConvLayer(384, 3, 1, 1), ReLU(), ConvLayer(384, 3, 1, 1), ReLU(), ConvLayer(256, 3, 1, 1), ReLU(), MaxPool(2, 2), Flatten(), DenseLayer(4096), ReLU(), DenseLayer(4096), ReLU(), DenseLayer(1000), Softmax()]
    
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
        loss = -np.sum(y_true * np.log(y_pred + 1e-10)) / y_pred.shape[0]
        return loss
    
    def f1_score(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        tp = np.sum(y_pred * y_true)
        fp = np.sum(y_pred * (1 - y_true))
        fn = np.sum((1 - y_pred) * y_true)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return f1
