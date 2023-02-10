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

# X_val = np.load('X_val.npy')
# y_val = pd.read_csv(os.path.join('..', 'dataset', 'training-d.csv'))
# y_val = y_val['digit'].values

# make X, y smaller 
# X, temp1, y, temp2 = train_test_split(X, y, test_size=0.9, random_state=42, stratify=y)

print(X.shape, y.shape)

#split train and validation : 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# X_train = X
# y_train = y

# one hot encoding
y = np.eye(10)[y_train]
y_val_true =  np.eye(10)[y_val]


#train 
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.7]

for learning_rate in learning_rates:
    model = Network(learning_rate)
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    training_f1 = []
    validation_f1 = []

    for i in range(1):
        print("Epoch %d" % i)
        for j in tqdm.tqdm(range(0,X.shape[0], 64)):
            #mini batch : choose 128 random images
            index = np.random.randint(0, X_train.shape[0], 64)
            X_batch = X_train[index]
            y_batch = y[index]
            model.train(X_batch, y_batch)

        y_pred = model.forward(X_train)


        #f1 score
        f1 = model.f1_score(y_pred, y)
        print("Training F1 Score = %.4f" % f1)
        training_f1.append(f1)

        #training loss 
        loss = model.cross_entropy_loss(y_pred, y)
        print("Training Loss = %.4f" % loss)
        training_loss.append(loss)

        #training accuracy
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y, axis=1)

        accuracy = np.sum(y_pred == y_true) / y_true.shape[0]
        accuracy = accuracy * 100
        print("Training Accuracy = %.4f" % accuracy)
        training_accuracy.append(accuracy)

    
        #validation

        y_pred = model.forward(X_val)

        #validation f1

        f1 = model.f1_score(y_pred, y_val_true)
        print("Validation F1 Score = %.4f" % f1)
        validation_f1.append(f1)

        #validation loss

        loss = model.cross_entropy_loss(y_pred, y_val_true)
        print("Validation Loss = %.4f" % loss)
        validation_loss.append(loss)

        #validation accuracy

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val_true, axis=1)

        accuracy = np.sum(y_pred == y_true) / y_true.shape[0]
        accuracy = accuracy * 100
        print("Validation Accuracy = %.4f" % accuracy)
        validation_accuracy.append(accuracy)

    pd.DataFrame({
        'Training Loss': training_loss,
        'Validation Loss': validation_loss,
        'Training Accuracy': training_accuracy,
        'Validation Accuracy': validation_accuracy,
        'Training F1': training_f1,
        'Validation F1': validation_f1
    }).to_csv('result_lr_'+str(learning_rate)+ '.csv', index=False )
    




