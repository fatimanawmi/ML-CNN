import numpy as np

class Softmax:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        #handling exploding gradients by clipping values with max value
        input = input - np.max(input, axis=1, keepdims=True)
        self.output = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        return self.output

    def backward(self, y):
        return self.output - y

    # def softmax_derivative(self, x):
    #     return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def clear(self):
        self.output = None
        self.input = None