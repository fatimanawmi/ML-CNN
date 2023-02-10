import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image1.png')
        image = image.astype(np.float32)
        image = image / 255.0
        image = 1 - image
        image = cv2.dilate(image, kernel, iterations=1)
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image2.png')

        blank, image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image3.png')

        #crop the image and make it 28 X 28
        image = image[2:30, 2:30]
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image4.png')
        
        # add a dimension to the image to make it a 3D array (i.e. a single image) instead of a 2D array (i.e. a list of images), make it 1 X 28 X 28 
        image = np.expand_dims(image, axis=0)
        # print(image.shape)
        images.append(image)

    # Combine the images into a single 4D numpy array
    images = np.array(images)


    return images




