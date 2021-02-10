# Vehicle Detection and Tracking
## Final Results via Deep Learning
Click to play video:

[![YOLOv5 Video](v4/0000000005.png)](https://youtu.be/sitoSZPr8HQ "YOLOv5 Video")

## 1st Implementation: Manual Feature Extraction + Classifier
1. Color Features
    - Color Histogram
    - Color Spaces (HSV, LUV, etc.)
    - Spatial Binning (`cv2.resize()`)

2. Gradient Features
    - Histogram of Oriented Gradients (HOG)

### Feature Extraction
- Normalize each feature, then combine into one feature vector
- Train-Test split
- Could even use a decision tree for feature selection
- Be careful of time dependencies, even w/ a random train-test split

### Train Vehicle Classifier
Train SVM to classify Car/Not Car.

### Sliding Window
Slide a window (at different scales) over the frames and for each window, classify car/not-car.

### Manual Feature Extraction Results
Click to play video:

[![Manual Feature Extraction Video](v3/0000000233.png)](https://youtu.be/N94dRl46f8k "Manual Feature Extraction Video")

## Deep Learning Implementation (YOLOv5)
Learning the features directly. 

Click to play video:

[![YOLOv5 Video](v4/0000000005.png)](https://youtu.be/sitoSZPr8HQ "YOLOv5 Video")