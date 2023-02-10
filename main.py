import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
import os 

from denseLayer import DenseLayer
from network import Network


X = np.load('X.npy')
path = os.path.join('..', 'dataset', 'training-a.csv')
y_a = pd.read_csv(path)
path = os.path.join('..', 'dataset', 'training-b.csv')
y_b = pd.read_csv(path)
path = os.path.join('..', 'dataset', 'training-c.csv')
y_c = pd.read_csv(path)
y = pd.concat([y_a, y_b, y_c])
y = y['digit'].values

X_val = np.load('X_val.npy')
y_val = pd.read_csv(os.path.join('..', 'dataset', 'training-d.csv'))
y_val = y_val['digit'].values

# make X, y smaller to 500 images
# X, X_val, y, y_val = train_test_split(X, y, test_size=0.9, random_state=42, stratify=y)

print(X.shape, y.shape)

#split train and validation : 80% train, 20% validation
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train = X
y_train = y

# one hot encoding
y = np.eye(10)[y_train]
y_val_true =  np.eye(10)[y_val]


#train 
model = Network(0.001)


for i in range(20):
    print("Epoch %d" % i)
    for j in tqdm.tqdm(range(0,X.shape[0], 64)):
        #mini batch : choose 128 random images
        index = np.random.randint(0, X_train.shape[0], 64)
        X_batch = X_train[index]
        y_batch = y[index]
        model.train(X_batch, y_batch)

    #training loss 
    y_pred = model.forward(X_train)
    loss = model.loss(y_pred, y)
    print("Training Loss = %.4f" % loss)

    #training accuracy
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val_true, axis=1)

    accuracy = np.sum(y_pred == y_true) / y_true.shape[0]
    accuracy = accuracy * 100
    print("Training Accuracy = %.4f" % accuracy)

    y_pred = model.forward(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val_true, axis=1)

    accuracy = np.sum(y_pred == y_true) / y_true.shape[0]
    accuracy = accuracy * 100
    print("Validation Accuracy = %.4f" % accuracy)

    loss = model.loss(y_pred, y_val_true)
    print("Validation Loss = %.4f" % loss)

    f1 = model.f1_score(y_pred, y_val_true)
    print("Validation F1 Score = %.4f" % f1)




