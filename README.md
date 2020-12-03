# Vehicle Detection and Tracking
Vehicle detection with computer vision

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

### Classifier
Starting w/ SVM, then try random forest, NN, or an ensemble

### Hyperparameter Tuning
`GridSearchCV` and/or `RandomizedSearchCV`

### Sliding Window
Slide a window (at different scales) over the frames and for each window, classify car/not-car.

## Deep Learning: Learn The Features Directly
Using YOLOv5