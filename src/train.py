import glob 
import os
import time

import cv2
import numpy as np
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from preprocessing import load_data, augment_images
import feature_extraction as feat

# load images
vehicles = load_data('../data/vehicles')
non_vehicles = load_data('../data/non-vehicles')

# data augmentation
vehicles = augment_images(vehicles)
non_vehicles = augment_images(non_vehicles)

# extract features from images
print('\nExtracting features...\n')
color_space = 'YCrCb'
spatial_size = (16, 16)
hist_bins = 24
start = time.time()
vehicles_features = feat.extract_features(
    vehicles, hist_bins=hist_bins, color_space=color_space, spatial_size=spatial_size
)
non_vehicles_features = feat.extract_features(
    non_vehicles, hist_bins=hist_bins, color_space=color_space, spatial_size=spatial_size
)
end = time.time()
print(int(end-start), 'seconds to extract features\n')

# Get train-test data, as well as X and y's (labels)
X = np.vstack((vehicles_features, non_vehicles_features))

# vehicles are 1, non-vehicles 0
vehicles_lables = np.ones(len(vehicles_features))
non_vehicles_labels = np.zeros(len(non_vehicles_features))
y = np.concatenate(vehicles_lables, non_vehicles_labels)
assert X.shape[0] == len(y)

# random train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42
)
print('Shapes before dim reduction:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# subsample
# half_idx = len(X_train) // 2
# X_train = X_train[:half_idx]
# y_train = y_train[:half_idx]
# half_idx = len(X_test) // 2
# X_test = X_test[:half_idx]
# y_test = y_test[:half_idx]

# normalize features (only fit on x_train to avoid peeking at test data)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
pickle.dump(scaler, open('model/scaler.p', 'wb'))
print('\nX_train after scale:', X_train[0])

# dimension reduction (only fit on x_train to avoid peeking at test data)
pca = PCA(n_components=200)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print('Dims after reduction:',  X_train.shape, y_train.shape, X_test.shape, y_test.shape)
pickle.dump(pca, open('model/pca.p', 'wb'))

# train
print('\nTraining...\n')
# prob=True is ~5x slower, does 5-fold CV but allows access to prob, not just pred (0/1)
svm = SVC(probability=True, verbose=100)
start = time.time()
svm.fit(X_train, y_train)
end = time.time()
print(int(end-start), 'seconds to train SVM\n')
pickle.dump(svm, open('model/svm.sav', 'wb'))

# evaluate
print('Test Accuracy:', svm.score(X_test, y_test))



