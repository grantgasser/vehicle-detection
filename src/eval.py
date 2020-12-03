import os

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

import feature_extraction as feat

# load save model, scaler, and pca
loaded_model = pickle.load(open('model/svm.sav', 'rb'))
scaler = pickle.load(open('model/scaler.p', 'rb'))
pca = pickle.load(open('model/pca.p', 'rb'))

# print(X_test.shape, y_test.shape)
holdout_vehicles = []
folder_path = '../data/holdout_vehicles'
for file in os.listdir(folder_path):
    if '.png' in file:
        # NOTE: opencv reads image as BGR, not RGB
        img = cv2.imread(os.path.join(folder_path, file))
        holdout_vehicles.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

holdout_non_vehicles = []
folder_path = '../data/holdout_non_vehicles'
for file in os.listdir(folder_path):
    if '.png' in file:
        # NOTE: opencv reads image as BGR, not RGB
        img = cv2.imread(os.path.join(folder_path, file))
        holdout_non_vehicles.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# feature extraction
color_space = 'YCrCb'
spatial_size = (16, 16)
hist_bins = 24
vehicles_features = feat.extract_features(
    holdout_vehicles, hist_bins=hist_bins, color_space=color_space, spatial_size=spatial_size
)
non_vehicles_features = feat.extract_features(
    holdout_non_vehicles, hist_bins=hist_bins, color_space=color_space, spatial_size=spatial_size
)

# predict vehicles
for i, feature in enumerate(vehicles_features):
    feature = feature.reshape(1, -1)
    feature = scaler.transform(feature)
    feature = pca.transform(feature)
    pred = loaded_model.predict(feature)

    pred = 'car' if pred else 'not car'
    plt.title('Predicted: {}. Actual: car'.format(pred))
    plt.imshow(holdout_vehicles[i])
    plt.show()


for i, feature in enumerate(non_vehicles_features):
    feature = feature.reshape(1, -1)
    feature = scaler.transform(feature)
    feature = pca.transform(feature)
    pred = loaded_model.predict(feature)

    pred = 'car' if pred else 'not car'
    plt.title('Predicted: {}. Actual: not car'.format(pred))
    plt.imshow(holdout_non_vehicles[i])
    plt.show()