import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
import os 
from network import Network
from matplotlib import pyplot as plt
import seaborn as sn
import pickle   


X = np.load('X.npy')
path = os.path.join('..', 'dataset', 'training-a.csv')
y_a = pd.read_csv(path)
path = os.path.join('..', 'dataset', 'training-b.csv')
y_b = pd.read_csv(path)
path = os.path.join('..', 'dataset', 'training-c.csv')
y_c = pd.read_csv(path)
y = pd.concat([y_a, y_b, y_c])
y = y['digit'].values

X_test = np.load('X_test.npy')
y_test = pd.read_csv(os.path.join('..', 'dataset', 'training-d.csv'))
y_test = y_test['digit'].values



print(X.shape, y.shape)


X_train = X
y_train = y

# one hot encoding
y = np.eye(10)[y_train]
y_test_true =  np.eye(10)[y_test]


#train 
learning_rate = 0.001
epoch = 20

model = Network(learning_rate)
training_loss = []
Test_loss = []
training_accuracy = []
Test_accuracy = []
training_f1 = []
Test_f1 = []

y_pred = None

for i in range(epoch):
    print("Epoch : ", i+1, "Learning Rate : ", learning_rate)
    for j in tqdm.tqdm(range(0,X.shape[0], 64)):
        #mini batch : choose 128 random images
        index = np.random.randint(0, X_train.shape[0], 64)
        X_batch = X_train[index]
        y_batch = y[index]
        model.train(X_batch, y_batch)

    #Test

    y_pred = model.forward(X_test)

    #Test loss

    loss = model.cross_entropy_loss(y_pred, y_test_true)
    print("Test Loss = %.4f" % loss)
    Test_loss.append(loss)

    #Test f1

    f1 = model.f1_macro(y_pred, y_test_true)
    print("Test F1 Score = %.4f" % f1)
    Test_f1.append(f1)


    #Test accuracy

    accuracy = model.accuracy(y_pred, y_test_true) * 100
    print("Test Accuracy = %.4f" % accuracy)
    Test_accuracy.append(accuracy)
    
    

#Test

y_pred = model.forward(X_test)

#Test loss

loss = model.cross_entropy_loss(y_pred, y_test_true)
print("Test Loss = %.4f" % loss)
Test_loss.append(loss)

#Test f1

f1 = model.f1_macro(y_pred, y_test_true)
print("Test F1 Score = %.4f" % f1)
Test_f1.append(f1)


#Test accuracy

accuracy = model.accuracy(y_pred, y_test_true) * 100
print("Test Accuracy = %.4f" % accuracy)
Test_accuracy.append(accuracy)

pd.DataFrame({
    'Test Loss': Test_loss,
    'Test Accuracy': Test_accuracy,
    'Test F1': Test_f1
}).to_csv('result_lr_'+str(learning_rate)+ '_.csv', index=False )

confusion_matrix = np.zeros((10, 10))

# generate confusion matrix : for ith row, jth column is the number of images of class i that are classified as class j
y_pred = np.argmax(y_pred, axis=1)
for i in range(10):
    for j in range(10):
        confusion_matrix[i][j] = np.sum(np.logical_and(y_pred == i, y_test == j))

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
df_cfm = pd.DataFrame(confusion_matrix, index = classes, columns = classes)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True, cmap="YlGnBu")
cfm_plot.figure.savefig("cfm_"+str(learning_rate)+ "_.png")


model.clear()
pickle.dump(model, open('1705093_model.pkl', 'wb'))

    




