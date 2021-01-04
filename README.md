# Vehicle Detection and Tracking
## Final Results
Click to play clip:

[![YOLOv5 Video](https://i9.ytimg.com/vi/sitoSZPr8HQ/mq1.jpg?sqp=CISUzv8F&rs=AOn4CLBQI4lPXqDDSeHQTGLNMD2pNIqzuA)](https://youtu.be/sitoSZPr8HQ "YOLOv5 Video")

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
Click to play:

[![Manual Feature Extraction Video](https://i9.ytimg.com/vi/N94dRl46f8k/mq2.jpg?sqp=CNyYzv8F&rs=AOn4CLCB51cnQ4zh0pTPqbUynmH0WOxFFQ)](https://youtu.be/N94dRl46f8k "Manual Feature Extraction Video")

## Deep Learning Implementation (YOLOv5)
Learning the features directly. 

Click to play:

[![YOLOv5 Video](https://i9.ytimg.com/vi/sitoSZPr8HQ/mq1.jpg?sqp=CISUzv8F&rs=AOn4CLBQI4lPXqDDSeHQTGLNMD2pNIqzuA)](https://youtu.be/sitoSZPr8HQ "YOLOv5 Video")