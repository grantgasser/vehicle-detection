from tqdm import tqdm
import numpy as np
import cv2
from skimage.feature import hog


def get_color_hist_features(img, bins=32, range=(0, 256)):
    """Compute and return color histogram features"""
    rhist = np.histogram(img[:, :, 0], bins=bins, range=range)
    ghist = np.histogram(img[:, :, 1], bins=bins, range=range)
    bhist = np.histogram(img[:, :, 2], bins=bins, range=range)

    features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    return features


def change_color_space(img, color_space='RGB'):
    """Changes color space and returns as feature"""
    if color_space == 'RGB':
        return img

    else:
        if color_space == 'LUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YCrCb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        else:
            raise ValueError('Invalid color space:', color_space)

    return img


def get_spatial_features(img, size=(32, 32)):
    """Performs spatial binning, reducing size of image, and unravel into feature vector"""
    spatial_features = cv2.resize(img, size).ravel()
    return spatial_features


def get_hog_features(img):
    """Get histogram of oriented gradients (HOG) features"""
    features = hog(img)
    return features


def extract_features(images, hist_bins, color_space, spatial_size):
    """
    Get feature vector for each image and store in a list.
    
    Increase param list if want to tweak and play around more.
    """
    features = []

    # get feature vec for each image
    for img in images:
        img = change_color_space(img, color_space=color_space)

        # features
        hist_features = get_color_hist_features(img, bins=hist_bins)
        spatial_features = get_spatial_features(img, size=spatial_size)
        hog_features = get_hog_features(img)

        # combine and store features for this image
        feature_vector = np.concatenate((hist_features, spatial_features, hog_features))
        features.append(feature_vector)

    return features