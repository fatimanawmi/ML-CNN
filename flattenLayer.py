import numpy as np

class Flatten:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = input.reshape(input.shape[0], -1)
        return self.output

    def backward(self, output_gradient):
        return  output_gradient.reshape(self.input.shape)
    
    def clear(self):
        pass