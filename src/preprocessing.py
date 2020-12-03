import os

import cv2
import numpy as np
from tqdm import tqdm
from skimage.filters import gaussian
from skimage.transform import warp, AffineTransform, rotate
from skimage.util import random_noise
import matplotlib.pyplot as plt

def load_data(base_path):
    """Load images of vehicles and non-vehicles from subfolders"""
    images = []

    # several folders of images
    for folder in tqdm(os.listdir(base_path), desc='Loading images'):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            # read each image in the folder and store
            for file in os.listdir(folder_path):
                if '.png' in file:
                    # NOTE: opencv reads image as BGR, not RGB
                    img = cv2.imread(os.path.join(folder_path, file))
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # print some stats
    print('Loaded {} {} images of type: {}\n'.format(len(images), images[0].shape, images[0].dtype))

    return images

# def kitti_to_yolo(base_path):
#     """
#     Read kitti images and their labels and convert to YOLO format.
    
#     Kitti: kitti_single/
#                 testing/
#                     image_2/
#                         000000.png
#                 training/
#                     image_2/
#                         000000.png
#                     label_2/
#                         000000.txt

#     where label file 000000.txt could look like:
#         Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
#         Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
#         Cyclist 0.00 3 -1.65 676.60 163.95 688.98 193.93 1.86 0.60 2.02 4.59 1.32 45.84 -1.55

#     to YOLO format with :
#         images/000000.png
#         labels/000000.txt

#     where label file 000000.txt looks like normalized 0-1 (class x_center y_center width height):
#         0 .3 .4 .77 .47
#     """
#     test_images = []
#     train_images = []

#     # from kitti
#     train_images_path = os.path.join(base_path, 'kitti_labelled/kitti_single/training/image_2/')
#     train_labels_path = os.path.join(base_path, 'kitti_labelled/kitti_single/training/label_2/')

#     # get training data
#     num_train_images = len(os.listdir(train_images_path))
#     val_thresh = int(0.9 * num_train_images)  # 10% validation data
#     image_idx = 0
#     for image_idx in tqdm(range(num_train_images), desc='Converting Kitti to YOLO'):
#         # get paths
#         file_str = str(image_idx).zfill(6)
#         image_path = os.path.join(train_images_path, file_str + '.png')
#         label_path = os.path.join(train_labels_path, file_str + '.txt')

#         # read image and labels
#         image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#         with open(label_path) as f:
#             lines = f.readlines()
        
#         labels = []
#         for line in lines:
#             cols = line.split()

#             if cols[0] == 'Truck' or cols[0] == 'Car' or cols[0] == 'Van':
#                 top_left_x = int(float(cols[4]))
#                 top_left_y = int(float(cols[5]))
#                 bottom_right_x = int(float(cols[6]))
#                 bottom_right_y = int(float(cols[7]))

#                 x_center = np.float32(((bottom_right_x + top_left_x) // 2) / image.shape[1])
#                 y_center = np.float32(((bottom_right_y + top_left_y) // 2) / image.shape[0])
#                 width = np.float32((bottom_right_x - top_left_x) / image.shape[1])
#                 height = np.float32((bottom_right_y - top_left_y) / image.shape[0])

#                 # store to write later
#                 labels.append([0, x_center, y_center, width, height])

#         # should we also use frames w/o a vehicle?
#         if labels:
#             # write image
#             if image_idx < val_thresh:
#                 image_path = base_path + 'kitti_to_yolo/images/train/'
#                 label_path = base_path + 'kitti_to_yolo/labels/train/'
#             else:
#                 image_path = base_path + 'kitti_to_yolo/images/val/'
#                 label_path = base_path + 'kitti_to_yolo/labels/val/'
#             cv2.imwrite(image_path + file_str + '.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

#             # write label
#             with open(label_path + file_str + '.txt', 'w') as f:
#                 for label in labels:
#                     str_label = [str(l) for l in label]

#                     for item in str_label:
#                         f.write(item + ' ')
#                     f.write('\n')

#         if image_idx == 10:
#             break



def augment_images(images):
    """
    For each image in images, apply several augmentations to create more, 
    diverse training data.

    NOTE: multiplicative increase in training data size
    """
    augmented_images = []
    for img in tqdm(images, desc='Augmenting Images'):
        augmented_images.append(img)  # original
        augmented_images.append(np.uint8(255*gaussian(img, sigma=1.5, multichannel=True)))  # blur
        tfm = AffineTransform(translation=(np.random.randint(-10, 10), np.random.randint(-10, 10)))
        augmented_images.append(np.uint8(255*warp(img, tfm, mode='wrap')))  # shift
        augmented_images.append(np.uint8(255*rotate(img, angle=np.random.randint(5, 20), mode='wrap')))  # rotate left
        augmented_images.append(np.uint8(255*rotate(img, angle=np.random.randint(-20, -5), mode='wrap')))  # rotate right
        augmented_images.append(np.uint8(255*random_noise(img, var=0.005)))  # noise
        augmented_images.append(np.fliplr(img))  # flip

    return augmented_images
        
