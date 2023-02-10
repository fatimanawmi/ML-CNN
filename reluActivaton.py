import numpy as np

class ReLU:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, output_gradient):
        input_gradient =  np.where(self.input > 0, output_gradient, 0)
        return input_gradient

    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x