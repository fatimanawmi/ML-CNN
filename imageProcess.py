import numpy as np
import os
from input import process_image, process_image2

#take numpy array as input and save them in a binary file

X_a = process_image2(os.path.join('..', 'dataset', 'training-a'))
X_b = process_image2(os.path.join('..', 'dataset', 'training-b'))
X_c = process_image2(os.path.join('..', 'dataset', 'training-c'))

X = np.concatenate((X_a, X_b, X_c), axis=0)

X_val = process_image2(os.path.join('..', 'dataset', 'training-d'))

np.save('X.npy', X)

np.save('X_val.npy', X_val)