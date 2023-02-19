# write a separate python script that can load the pickle file of your trained model and use 
# it to predict labels for query images (i.e., the images for which classification needs to be done) from a specific folder. The path of 
# the folder will be a command line parameter. In the same folder you will prepare a CSV file with 2 columns. 
# The first column contains the input file names (just the name, excluding path) and the second column contains the corresponding predicted digit.


import sys
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Train_1705093 import *



def process_image(path):
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

    # Load and process each image
    images = []
    names = []
    for file in image_files:
        print('Processing image: %s' % file)
        names.append(file)
        
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


    return images, names

#take path as input from command line

path = sys.argv[1]

model = pickle.load(open('1705093_model.pkl', 'rb'))

X, names = process_image(path)

y_pred = model.forward(X)

y_pred = np.argmax(y_pred, axis=1)

#make a dataframe with 2 columns, first column contains the input file names (just the name, excluding path) and the second column contains the corresponding predicted digit.

df = pd.DataFrame({'FileName': names, 'Digit': y_pred})

df.to_csv(path + '/1705093_prediction.csv', index=False)



