#write a class for maxpool layer with forward and backward pass, it will take a. Filter dimension
# b. Stride as input 

import numpy as np

class MaxPool:
    def __init__(self, filter_dim, stride):
        self.filter_dim = filter_dim
        self.stride = stride
        self.input = None
        self.left_gradient = None
        self.out_h = None
        self.out_w = None
        self.mask = None

    def forward(self, input):
        self.input = input

        #output dimensions
        self.out_h = int(((input.shape[2] - self.filter_dim) / self.stride) + 1)
        self.out_w = int(((input.shape[3] - self.filter_dim) / self.stride) + 1)

        #shape for strided input --> (batch_size, channels, out_h, out_w, filter_dim, filter_dim)
        strided_input = np.lib.stride_tricks.as_strided(input,
                                                        shape=(input.shape[0], input.shape[1], self.out_h, self.out_w, self.filter_dim, self.filter_dim),
                                                        strides=(input.strides[0], input.strides[1], input.strides[2] * self.stride, input.strides[3] * self.stride, input.strides[2], input.strides[3]))
        
        output = np.max(strided_input, axis=(4, 5))
        
        # repeat the max value in the filter_dim x filter_dim window
        repeated_max = output.repeat(self.filter_dim, axis=2).repeat(self.filter_dim, axis=3)


        cropped_input = input[:, :, :self.out_h * self.stride, :self.out_w * self.stride]
        self.mask = np.equal(cropped_input, repeated_max).astype(int)

        return output

    def backward(self, right_gradient):
        if self.stride == self.filter_dim:
            self.left_gradient = right_gradient.repeat(self.filter_dim, axis=2).repeat(self.filter_dim, axis=3)
            self.left_gradient = np.multiply(self.left_gradient, self.mask)
            padded = np.zeros(self.input.shape)
            padded[:, :, :self.left_gradient.shape[2], :self.left_gradient.shape[3]] = self.left_gradient
            return padded
        else:
            # self.left_gradient = np.zeros(self.input.shape)
            # for i in range(self.out_h):
            #     for j in range(self.out_w):
            #         self.left_gradient[:, :, i*self.stride:i*self.stride+self.filter_dim, j*self.stride:j*self.stride+self.filter_dim] += np.multiply(self.mask[:, :, i, j], right_gradient[:, :, i, j])
            # return self.left_gradient
            pass

    def clear(self):
        self.left_gradient = None
    