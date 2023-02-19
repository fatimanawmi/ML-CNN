import numpy as np
import pandas as pd
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import log_loss

import os 
from matplotlib import pyplot as plt
import seaborn as sn
import cv2
import pickle


#=======================================================================================================================================================

#                                                          Softmax:


#=======================================================================================================================================================

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
        
        
#=======================================================================================================================================================

#                                                          Flatten

#=======================================================================================================================================================

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
    
#=======================================================================================================================================================

#                                                          ReLU

#=======================================================================================================================================================
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
    
    def clear(self):
        self.input = None
        self.output = None

#=======================================================================================================================================================

#                                                          MaxPool

#=======================================================================================================================================================

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
                                                        strides=(input.strides[0], input.strides[1], input.strides[2] * self.stride, 
                                                                 input.strides[3] * self.stride, input.strides[2], input.strides[3]))
        
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
            pass

    def clear(self):
        self.left_gradient = None



#=======================================================================================================================================================

#                                                          DenseLayer

#=======================================================================================================================================================

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

#=======================================================================================================================================================

#                                                          ConvLayer

#=======================================================================================================================================================
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
                                                        strides=(padded_input.strides[0], padded_input.strides[1],
                                                                 padded_input.strides[2] * self.stride, 
                                                                 padded_input.strides[3] * self.stride, 
                                                                 padded_input.strides[2], padded_input.strides[3]))
        
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
        dilated_right_gradient = np.zeros((batch_size, right_gradient.shape[1], right_gradient.shape[2] + (right_gradient.shape[2] - 1) * (self.stride - 1), 
                                           right_gradient.shape[3] + (right_gradient.shape[3] - 1) * (self.stride - 1)))
        dilated_right_gradient[:, :, ::self.stride, ::self.stride] = right_gradient



        # gradient of filters = convolution of padded input and dilated_right_gradient
        # convolution of padded input and dilated_right_gradient : using strided input
        # shape of strided input --> (batch_size, num_filters, out_h, out_w, filter_dim, filter_dim)
        strided_input = np.lib.stride_tricks.as_strided(padded_input, 
                                                        shape=(padded_input.shape[0], padded_input.shape[1], self.filter_dim, 
                                                               self.filter_dim, dilated_right_gradient.shape[2], dilated_right_gradient.shape[3]), 
                                                        strides=(padded_input.strides[0], padded_input.strides[1], padded_input.strides[2], 
                                                                 padded_input.strides[3], padded_input.strides[2], padded_input.strides[3]))
        
        self.filters_gradient = np.einsum('bchwkl, bnkl -> nchw', strided_input, dilated_right_gradient)

        # padding for dilated_right_gradient to match the size of padded_input : dilate amount -> (filter_dim - 1)
        padded_dilated_right_gradient = np.pad(dilated_right_gradient, ((0, 0), (0, 0), (self.filter_dim - 1, self.filter_dim - 1), 
                                                                        (self.filter_dim - 1, self.filter_dim - 1)), 'constant')
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
        # assert not np.isnan(self.filters_gradient).any()
        # assert not np.isnan(self.bias_gradient).any()
        # assert not np.isinf(self.filters_gradient).any()
        # assert not np.isinf(self.bias_gradient).any()
        # assert not np.isnan(self.left_gradient).any()
        # assert not np.isinf(self.left_gradient).any()
        # assert not np.isnan(self.filters).any()
        # assert not np.isnan(self.bias).any()
        # assert not np.isinf(self.filters).any()
        # assert not np.isinf(self.bias).any()

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
        


#=======================================================================================================================================================

#                                                          Network

#=======================================================================================================================================================
class Network:
    def __init__(self, learning_rate):
        # lenet
        self.model = [ConvLayer(6, 5, 1, 0,learning_rate), ReLU(), MaxPool(2, 2), ConvLayer(16, 5, 1, 0,learning_rate), ReLU(), 
                      MaxPool(2, 2), Flatten(), DenseLayer(120,learning_rate), ReLU(), DenseLayer(84,learning_rate), ReLU(), DenseLayer(10,learning_rate), Softmax()]
    
    def forward(self, X):
        for layer in self.model:
            # print(X.shape, layer)
            X = layer.forward(X)
            # if np.isnan(X).any() or np.isinf(X).any():
            #     print("X is nan or inf")
            #     print(X)
            #     print(layer)
            #     exit()
        return X

    def backward(self, y):
        grad = y
        for layer in reversed(self.model):
            # print if grad is nan or inf
            # if np.isnan(grad).any() or np.isinf(grad).any():
            #     print("grad is nan or inf")
            #     print(grad)
            #     print(layer)
            #     exit()
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
            
    def pickle_predict(self, X):
        y_pred = self.forward(X)
        return y_pred

#=======================================================================================================================================================

#                                                          process_image

#=======================================================================================================================================================


def process_image(path):
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

    # Load and process each image
    images = []
    for file in image_files:
        print('Processing image: %s' % file)
        
        image = cv2.imread(os.path.join(path, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))
        kernel = np.ones((3, 3), np.uint8)
       
        image = image.astype(np.float32)
        image = image / 255.0
        image = 1 - image
        image = cv2.dilate(image, kernel, iterations=1)
        
        blank, image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
        

        #crop the image and make it 28 X 28
        image = image[2:30, 2:30]
       
        
        # add a dimension to the image to make it a 3D array (i.e. a single image) instead of a 2D array (i.e. a list of images), make it 1 X 28 X 28 
        image = np.expand_dims(image, axis=0)
        images.append(image)

    # Combine the images into a single 4D numpy array
    images = np.array(images)


    return images

#=======================================================================================================================================================

#                                                          get_X_y

#=======================================================================================================================================================

def get_X_y():
    
    # X_a = process_image(os.path.join('..', 'dataset', 'training-a'))
    # X_b = process_image(os.path.join('..', 'dataset', 'training-b'))
    # X_c = process_image(os.path.join('..', 'dataset', 'training-c'))

    # X = np.concatenate((X_a, X_b, X_c), axis=0)

    # X_val = process_image(os.path.join('..', 'dataset', 'training-d'))

    # np.save('X.npy', X)

    # np.save('X_val.npy', X_val)

    X_val = np.load('X_test.npy')
    y_val = pd.read_csv(os.path.join('..', 'dataset', 'training-d.csv'))
    y_val = y_val['digit'].values

    X = np.load('X.npy')
    path = os.path.join('..', 'dataset', 'training-a.csv')
    y_a = pd.read_csv(path)
    path = os.path.join('..', 'dataset', 'training-b.csv')
    y_b = pd.read_csv(path)
    path = os.path.join('..', 'dataset', 'training-c.csv')
    y_c = pd.read_csv(path)
    y = pd.concat([y_a, y_b, y_c])
    y = y['digit'].values
    
    # X = X_a
    # y = y_a['digit'].values
    
    return X, y, X_val, y_val


#=======================================================================================================================================================

#                                                          main

#=======================================================================================================================================================

X, y, X_val, y_val= get_X_y()
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# one hot encoding
y = np.eye(10)[y]
y_val_true =  np.eye(10)[y_val]

X_train = X
y_train = y


learning_rate = 0.001
epoch = 25

model = Network(learning_rate)

y_pred = None

training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []
training_f1 = []
validation_f1 = []


for i in range(epoch):
    print("Epoch : ", i+1, "Learning Rate : ", learning_rate)
    for j in tqdm.tqdm(range(0,X.shape[0], 64)):
        #mini batch : choose 128 random images
        index = np.random.randint(0, X_train.shape[0], 64)
        X_batch = X_train[index]
        y_batch = y[index]
        model.train(X_batch, y_batch)

    y_pred = model.forward(X_train)

    #training loss 
    loss = model.cross_entropy_loss(y_pred, y)
    print("Training Loss = %.4f" % loss)
    training_loss.append(loss)

    #f1 score
    f1 = model.f1_macro(y_pred, y)
    print("Training F1 Score = %.4f" % f1)
    training_f1.append(f1)

    
    #training accuracy

    accuracy = model.accuracy(y_pred, y) * 100
    print("Training Accuracy = %.4f" % accuracy)
    training_accuracy.append(accuracy)

    #validation

    y_pred = model.forward(X_val)

    #validation loss

    loss = model.cross_entropy_loss(y_pred, y_val_true)
    print("Validation Loss = %.4f" % loss)
    validation_loss.append(loss)

    #validation f1

    f1 = model.f1_macro(y_pred, y_val_true)
    print("Validation F1 Score = %.4f" % f1)
    validation_f1.append(f1)

    
    #validation accuracy

    accuracy = model.accuracy(y_pred, y_val_true) * 100
    print("Validation Accuracy = %.4f" % accuracy)
    validation_accuracy.append(accuracy)


pd.DataFrame({
    'Training Loss': training_loss,
    'Validation Loss': validation_loss,
    'Training Accuracy': training_accuracy,
    'Validation Accuracy': validation_accuracy,
    'Training F1': training_f1,
    'Validation F1': validation_f1
}).to_csv('result_lr_'+str(learning_rate)+ '_.csv', index=False )

# confusion_matrix = np.zeros((10, 10))

# # generate confusion matrix : for ith row, jth column is the number of images of class i that are classified as class j
# y_pred = np.argmax(y_pred, axis=1)
# for i in range(10):
#     for j in range(10):
#         confusion_matrix[i][j] = np.sum(np.logical_and(y_pred == i, y_val == j))

# classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# df_cfm = pd.DataFrame(confusion_matrix, index = classes, columns = classes)
# plt.figure(figsize = (10,7))
# cfm_plot = sn.heatmap(df_cfm, annot=True, cmap="YlGnBu")
# cfm_plot.figure.savefig("cfm_"+str(learning_rate)+ "_.png")


# model.clear()
# pickle.dump(model, open('1705093_model.pkl', 'wb'))