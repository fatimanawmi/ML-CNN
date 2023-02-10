#convolutional layer with forward and backward pass, it will take 
#  a. Number of output channels
# b. Filter dimension
# c. Stride
# d. Padding

import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_dim, stride, padding, learning_rate=0.001):
        self.num_filters = num_filters
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding
        self.input = None
        self.filters = None
        self.bias = np.zeros(num_filters)
        self.learning_rate = learning_rate
        self.out_h = None
        self.out_w = None

        # for adam optimizer
        self.iteration = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.filters_momentum = 0
        self.bias_momentum = 0

        self.filters_variance = 0
        self.bias_variance = 0

        self.filters_momentum_hat = 0
        self.bias_momentum_hat = 0

        self.filters_variance_hat = 0
        self.bias_variance_hat = 0

        self.l2_lambda = 0.001
        

    def forward(self, input):
        self.input = input

        if self.filters is None:
            self.filters = np.random.randn(self.num_filters, input.shape[1], self.filter_dim, self.filter_dim) * np.sqrt(2 / (self.num_filters * self.filter_dim * self.filter_dim))

        #output dimensions
        self.out_h = int(((input.shape[2] - self.filter_dim + 2 * self.padding) / self.stride) + 1)
        self.out_w = int(((input.shape[3] - self.filter_dim + 2 * self.padding) / self.stride) + 1)

        #padding
        padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        #shape for strided input --> (batch_size, num_filters, out_h, out_w, filter_dim, filter_dim)
        strided_input = np.lib.stride_tricks.as_strided(padded_input, 
                                                        shape=(padded_input.shape[0], padded_input.shape[1], self.out_h, self.out_w, self.filter_dim, self.filter_dim), 
                                                        strides=(padded_input.strides[0], padded_input.strides[1], padded_input.strides[2] * self.stride, padded_input.strides[3] * self.stride, padded_input.strides[2], padded_input.strides[3]))
        
        # print(strided_input.shape, self.input.shape, self.filters.shape, padded_input.shape)
        output = np.einsum('bchwkl, nckl -> bnhw', strided_input, self.filters) + self.bias.reshape(1, self.num_filters, 1, 1)

        return output
    
    def backward(self, right_gradient):
        #shape for filters gradient --> ( num_filters, channels, filter_dim, filter_dim)
        #shape for bias gradient --> (num_filters)
        #shape for left gradient --> (batch_size, channels, height, width)
        
        batch_size = self.input.shape[0]
        channels = self.input.shape[1]
        height = self.input.shape[2]
        width = self.input.shape[3]

        #padding
        padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        # padded_right_gradient = np.pad(right_gradient, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        # calculate bias gradient by summing over right_gradient : all the dimensions except the num_filters
        self.bias_gradient = np.sum(right_gradient, axis=(0, 2, 3))  

        #dilate right_gradient to match the size of input : dilate amount -> (stride - 1)
        dilated_right_gradient = np.zeros((batch_size, right_gradient.shape[1], right_gradient.shape[2] + (right_gradient.shape[2] - 1) * (self.stride - 1), right_gradient.shape[3] + (right_gradient.shape[3] - 1) * (self.stride - 1)))
        dilated_right_gradient[:, :, ::self.stride, ::self.stride] = right_gradient



        # gradient of filters = convolution of padded input and dilated_right_gradient
        # convolution of padded input and dilated_right_gradient : using strided input
        # shape of strided input --> (batch_size, num_filters, out_h, out_w, filter_dim, filter_dim)
        strided_input = np.lib.stride_tricks.as_strided(padded_input, 
                                                        shape=(padded_input.shape[0], padded_input.shape[1], self.filter_dim, self.filter_dim, dilated_right_gradient.shape[2], dilated_right_gradient.shape[3]), 
                                                        strides=(padded_input.strides[0], padded_input.strides[1], padded_input.strides[2], padded_input.strides[3], padded_input.strides[2], padded_input.strides[3]))
        
        self.filters_gradient = np.einsum('bchwkl, bnkl -> nchw', strided_input, dilated_right_gradient)

        # padding for dilated_right_gradient to match the size of padded_input : dilate amount -> (filter_dim - 1)
        padded_dilated_right_gradient = np.pad(dilated_right_gradient, ((0, 0), (0, 0), (self.filter_dim - 1, self.filter_dim - 1), (self.filter_dim - 1, self.filter_dim - 1)), 'constant')
        # rotate filters by 180 degrees
        rotated_filters = np.rot90(self.filters, 2, (2, 3))
        # gradient of input = convolution of padded_dilated_right_gradient and rotated_filters
        padded_dilated_right_gradient_strided = np.lib.stride_tricks.as_strided(padded_dilated_right_gradient,
                                                                                shape=(padded_dilated_right_gradient.shape[0],
                                                                                        padded_dilated_right_gradient.shape[1],
                                                                                        padded_input.shape[2],
                                                                                        padded_input.shape[3],
                                                                                        self.filter_dim,
                                                                                        self.filter_dim),

                                                                                strides=(padded_dilated_right_gradient.strides[0],
                                                                                        padded_dilated_right_gradient.strides[1],
                                                                                        padded_dilated_right_gradient.strides[2],
                                                                                        padded_dilated_right_gradient.strides[3],
                                                                                        padded_dilated_right_gradient.strides[2],
                                                                                        padded_dilated_right_gradient.strides[3]))
        
        self.left_gradient = np.einsum('bnhwkl, nckl -> bchw', padded_dilated_right_gradient_strided, rotated_filters)
        self.left_gradient = self.left_gradient[:, :, self.padding : self.padding + height, self.padding : self.padding + width]
        self.update(self.learning_rate)

        return self.left_gradient
    
    def update(self, learning_rate):
        # adam optimizer and l2 regularization

        self.iteration += 1
        self.filters_gradient = self.filters_gradient / self.input.shape[0]
        self.bias_gradient = self.bias_gradient / self.input.shape[0]

        self.filters_gradient += self.l2_lambda * self.filters
        self.bias_gradient += self.l2_lambda * self.bias

        self.filters_momentum = self.beta1 * self.filters_momentum + (1 - self.beta1) * self.filters_gradient
        self.bias_momentum = self.beta1 * self.bias_momentum + (1 - self.beta1) * self.bias_gradient

        self.filters_variance = self.beta2 * self.filters_variance + (1 - self.beta2) * self.filters_gradient ** 2
        self.bias_variance = self.beta2 * self.bias_variance + (1 - self.beta2) * self.bias_gradient ** 2

        self.filters_momentum_hat = self.filters_momentum / (1 - self.beta1 ** self.iteration)
        self.bias_momentum_hat = self.bias_momentum / (1 - self.beta1 ** self.iteration)

        self.filters_variance_hat = self.filters_variance / (1 - self.beta2 ** self.iteration)
        self.bias_variance_hat = self.bias_variance / (1 - self.beta2 ** self.iteration)

        self.filters -= learning_rate * self.filters_momentum_hat / (np.sqrt(self.filters_variance_hat) + self.epsilon)
        self.bias -= learning_rate * self.bias_momentum_hat / (np.sqrt(self.bias_variance_hat) + self.epsilon)
        
        
        # self.filters -= learning_rate * self.filters_gradient * 1 / self.input.shape[0]
        # self.bias -= learning_rate * self.bias_gradient * 1 / self.input.shape[0]

        #check if the gradients are not nan or inf
        assert not np.isnan(self.filters_gradient).any()
        assert not np.isnan(self.bias_gradient).any()
        assert not np.isinf(self.filters_gradient).any()
        assert not np.isinf(self.bias_gradient).any()
        assert not np.isnan(self.left_gradient).any()
        assert not np.isinf(self.left_gradient).any()
        assert not np.isnan(self.filters).any()
        assert not np.isnan(self.bias).any()
        assert not np.isinf(self.filters).any()
        assert not np.isinf(self.bias).any()

    def clear(self):
        #clear every stored variable except for the ones used in forward pass
        self.left_gradient = None
        self.filters_gradient = None
        self.bias_gradient = None
        self.input = None
        self.output = None
        self.filters_momentum = None
        self.bias_momentum = None
        self.filters_variance = None
        self.bias_variance = None
        self.filters_momentum_hat = None
        self.bias_momentum_hat = None
        self.filters_variance_hat = None
        self.bias_variance_hat = None
        self.iteration = 0
        

