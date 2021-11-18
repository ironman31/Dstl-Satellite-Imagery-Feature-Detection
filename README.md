# Dstl-Satellite-Imagery-Feature-Detection
I have worked on Dstl Satellite Imagery Feature Detection which is an image segmentation problem. It's a Kaggle competition problem. The Data set is satellite images where is each image covers 1km sqr. The goal of the project is to identify buildings, rivers, roads like that which contain 10 classes.  The training data set includes 25 images, each with 20 channels (3 band (3 channels, RGB) + A band (8 channels) + M band (8 channels) + P band (1 channel)), and the corresponding labels of objects. There are 10 types of overlapping objects labeled with contours (wkt type of data). I have used Segnet model and got a 0.44 IOU score.