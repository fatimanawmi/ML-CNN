#class for dense layer with one parameter: output dimension

import numpy as np

class DenseLayer:
    def __init__(self, output_size, learning_rate=0.001):
        self.input_size = None
        self.output_size = output_size
        self.W = None
        self.b = np.random.randn(1, output_size)
        self.m = None
        self.X = None

        # adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.iteration = 0
        self.W_momentum = 0
        self.b_momentum = 0
        self.W_variance = 0
        self.b_variance = 0
        self.W_momentum_hat = 0
        self.b_momentum_hat = 0
        self.W_variance_hat = 0
        self.b_variance_hat = 0

        self.l2_lambda = 0.001
        
        self.learning_rate = learning_rate



    def forward(self, X):
        # print("X.shape", X.shape)
        # X has dimension m X channels X height X width
        if self.input_size is None:
            self.input_size = X.shape[1]
        if self.W is None:
             self.W = np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)
        self.m = X.shape[0]
        self.X = X
        self.z = np.dot(X, self.W) + self.b
        return self.z

    def cross_entropy_loss(self, y_pred, y_true):
        loss = -np.sum(y_true * np.log(y_pred + 1e-10)) / self.m
        return loss

    def backward(self, grad):
        dW = np.dot(self.X.T, grad) / self.m
        db = np.sum(grad, axis=0, keepdims=True) / self.m
        out_grad = np.dot(grad, self.W.T)
        self.update_weights(dW, db)
        return out_grad

    def update_weights(self, dW, db):
        # adam optimizer and l2 regularization
        self.iteration += 1
        dW += self.l2_lambda * self.W
        db += self.l2_lambda * self.b

        self.W_momentum = self.beta1 * self.W_momentum + (1 - self.beta1) * dW
        self.b_momentum = self.beta1 * self.b_momentum + (1 - self.beta1) * db
        self.W_variance = self.beta2 * self.W_variance + (1 - self.beta2) * dW**2
        self.b_variance = self.beta2 * self.b_variance + (1 - self.beta2) * db**2
        self.W_momentum_hat = self.W_momentum / (1 - self.beta1**self.iteration)
        self.b_momentum_hat = self.b_momentum / (1 - self.beta1**self.iteration)
        self.W_variance_hat = self.W_variance / (1 - self.beta2**self.iteration)
        self.b_variance_hat = self.b_variance / (1 - self.beta2**self.iteration)
        self.W -= self.learning_rate * self.W_momentum_hat / (np.sqrt(self.W_variance_hat) + self.epsilon)
        self.b -= self.learning_rate * self.b_momentum_hat / (np.sqrt(self.b_variance_hat) + self.epsilon)

        # self.W -= learning_rate * dW
        # self.b -= learning_rate * db

    def clear(self):
        self.m = None
        self.X = None
        self.z = None
        self.W_momentum = 0
        self.b_momentum = 0
        self.W_variance = 0
        self.b_variance = 0
        self.W_momentum_hat = 0
        self.b_momentum_hat = 0
        self.W_variance_hat = 0
        self.b_variance_hat = 0
        self.iteration = 0
        

