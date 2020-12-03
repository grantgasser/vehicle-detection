import os
import time

import cv2
import pickle
import numpy as np
from scipy.ndimage.measurements import label
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import feature_extraction as feat
from postprocessing import write_images, make_video

def slide_windows(img, x_start, x_stop, y_start, y_stop, window_size, overlap):
    """
    Apply sliding window given scale to image, store each one as subimage to 
    be later predicted car/not car.
    """
    windows = []
    
    # how much to shift window each step (e.g. 32 if window=64 and overlap=0.5)
    shift_x = window_size[0] * (1 - overlap[0])
    shift_y = window_size[1] * (1 - overlap[1])
    
    # length
    len_x = x_stop - x_start
    len_y = y_stop - y_start
    
    # number of windows in each direction
    windows_x = int((len_x - (window_size[0]*overlap[0]))/(shift_x))
    windows_y = int((len_y - (window_size[1]*overlap[1]))/(shift_y))
        
    windows = []
    for y in range(windows_y):
        for x in range(windows_x):
            # get top left and bottom right of window
            top_left = (int(x*shift_x + x_start), int(y*shift_y + y_start))
            bottom_right = (int(x*shift_x + x_start + window_size[0]), int(y*shift_y + y_start + window_size[1]))
            
            # append this window/box to the list 
            windows.append((top_left, bottom_right))
            
    return windows


def predict_boxes(frame, model, scaler, pca, color_space, spatial_size, hist_bins, base_window_size=(64,64), 
                  scales=[1, 2, 3, 4, 6], overlaps=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5),  (0.6, 0.6), (0.75, 0.75)],
                  confidence_thresh=0.98, verbose=False):
    # store windows where car was predicted
    car_boxes = []
    total_windows = 0
    total_pos = 0

    for scale, overlap in zip(scales, overlaps):
        # y_start_stop hard-coded for now, data-specific unfortunately, more related to camera mount positioning
        windows = slide_windows(
            frame, x_start=0, x_stop=frame.shape[1], y_start=200, y_stop=frame.shape[0], 
            window_size=np.array(base_window_size)*scale, overlap=overlap
        )

        # iterate thru windows
        for window in windows:
            total_windows += 1
            # feature extraction and classifier trained on base window size
            downsampled_img = cv2.resize(
                frame[window[0][1]:window[1][1], window[0][0]:window[1][0]], base_window_size
            )

            # get features
            features = feat.extract_features(
                [downsampled_img], hist_bins=hist_bins, color_space=color_space, spatial_size=spatial_size
            )
            
            # scale and dim reduction
            feature = features[0].reshape(1, -1)
            feature = scaler.transform(feature)
            feature = pca.transform(feature)

            # now predict
            pred = model.predict(feature)
            prob = model.predict_proba(feature)

            # require higher confidence to reduce false positives
            if prob[0][1] > confidence_thresh:
                car_boxes.append(window)
                total_pos += 1
    
    if verbose: 
        print('predicted pos/total = {}'.format(total_pos/total_windows, '%.2f'))
        
    return car_boxes, downsampled_img, features, pred    


def draw_bounding_boxes(img, labels):
    """
    Draws bounding boxes on img given labels (a tuple of form (labelled_img, num_boxes))

    e.g. (np.array([[0, 0, 1], [0, 0, 1]]), 1)
    """
    for box_num in range(1, labels[1] + 1):
        # locate label pixels, derive box
        arr = (box_num == labels[0]).nonzero()
        box = (arr[1].min(), arr[0].min()), (arr[1].max(), arr[0].max())

        # only accept if certain size, height, width > X
        h = box[1][0] - box[0][0]
        w = box[1][1] - box[0][1]
        if h >= 64 and w >= 64:
            cv2.rectangle(img, box[0], box[1], (0, 255, 0), 5)

        return img


def predict_frames(frames_path, loaded_model, scaler, pca, threshold, verbose=True):
    """
    Uses the classifier to predict bounding boxes for each frame in frames_path
        
    NOTE: expects frame/image file format: 0000000000.png
    """
    video_images = []
    image_idx = 1

    # read frames in order
    num_frames = len(os.listdir(frames_path))
    while image_idx < num_frames:
        file_str = str(image_idx).zfill(10)
        test_img = cv2.imread(frames_path + file_str + '.png')
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        start = time.time()
        car_boxes, downsampled_img, features, pred = predict_boxes(
            frame=test_img,
            model=loaded_model,
            scaler=scaler,
            pca=pca,
            color_space='YCrCb',
            spatial_size=(16,16),
            hist_bins=24,
            scales=[2, 3],
            overlaps=[(0.5, 0.5), (0.6, 0.6)],
            confidence_thresh=0.75,
            verbose=verbose
        )
        end = time.time()

        # filter out false positives
        heatmap = np.zeros_like(test_img[:, :, 0])

        for box in car_boxes:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # threshold
        heatmap[heatmap <= threshold] = 0
        labels = label(heatmap)

        # draw the bounding boxes
        draw_img = draw_bounding_boxes(test_img, labels)
        video_images.append(draw_img)
        
        if verbose:
            print(end-start, 'seconds to predict 1 frame.')
            plt.imshow(draw_img)
            plt.show()

        image_idx += 1
        
    return video_images
        

def main():
    # load trained model, scaler, and pca obj
    loaded_model = pickle.load(open('model/svm.sav', 'rb'))
    scaler = pickle.load(open('model/scaler.p', 'rb'))
    pca = pickle.load(open('model/pca.p', 'rb'))

    # get frames for prediction
    # frames_path = '../data/driving_data/2011_09_26/2011_09_26_drive_0014_sync/image_02/data/'
    frames_path = '../data/driving_data/2011_09_26-2/2011_09_26_drive_0059_extract/image_02/data/'  # 379 frames, 37 seconds
    threshold = 3
    video_images = predict_frames(frames_path, loaded_model, scaler, pca, threshold, verbose=False)

    # write images and make video
    video_images_files = write_images(video_images, '../test')
    make_video(video_images_files, fps=10)


main()