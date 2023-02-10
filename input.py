import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process_image2(path):

    image_files = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

    # Load and process each image
    images = []
    for file in image_files:
        print('Processing image: %s' % file)
        image = cv2.imread(os.path.join(path, file))
        
        #grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        image = adjust_contrast_grey(image)
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image1.png')
        
        # Invert image 
        image = 255 - image
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image2.png')
        
        
        
        # Apply dilation
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image3.png')
        
        # Apply Thresholding
        blank, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image4.png')
        
        # Apply cropping by dropping rows and columns with average pixel intensity >= 253
        # image  = np.where(image > 130, 0, image)
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image5.png')
        
        # Resize to 28x28
        image = cv2.resize(image, (28, 28))
        
        # Divide pixel values by 255
        image = image / 255.0
        # plt.imshow(image, cmap='gray')
        # plt.savefig('image6.png')
        
        image = np.expand_dims(image, axis=0)
        # print(image.shape)
        images.append(image)
        
        
     # Combine the images into a single 4D numpy array
    images = np.array(images)


    return images    
        
def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/(high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./(high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img
    

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

# def process_label(path):
#     # Load the labels
#     labels = pd.read_csv(path)

#     # Convert the labels into a 1D numpy array
#     labels = labels['digit'].values

#     return labels

# def train_validation_split(X,Y):
#     # Split the data into training and validation sets
#     split = int(0.8 * X.shape[0])
#     X_train = X[:split]
#     X_val = X[split:]
#     Y_train = Y[:split]
#     Y_val = Y[split:]

#     return X_train, X_val, Y_train, Y_val



