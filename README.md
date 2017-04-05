

Vehicle Detection Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image3]: ./output_images/test5_boxes.jpg
[image4]: ./output_images/test5_draw.jpg
[image8]: ./output_images/test6_boxes.jpg
[image9]: ./output_images/test6_draw.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


Writeup / README
---

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

Histogram of Oriented Gradients (HOG)
---

1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 105 through 116 and 29 through 46 of the file called `feature_extraction_utils.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

For each such set of parameters, I trained the classifier and tested it on 20% of the data which was left out of the training. Accuracy was highest when using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. 

2. Explain how you settled on your final choice of HOG parameters.

I chose the HOG parameters which produced the model with the highest accuracy (0.989). 

3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, color histogram, and spatial binning. The code can be found in the file `car_classifier.py` starting in line 27. Feature extraction functions can be found in the file `feature_extraction_utils.py`.

Sliding Window Search
---

1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This took some experimentation. It was a tradeoff between runtime (which played a large factor) and the number of overlapping windows, which improves detection accuracy. I finally decided to use `1, 1.5, 2, 3` as the scales of window sizes.
![alt text][image3]

2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a relatively accurate classifier. I also thresholded the heatmap for anything below 3 hot boxes, in order to remove false detections. Here are some example images:

Hot box detections:
![alt text][image3]
after thresholding:
![alt text][image4]
Hot box detections:
![alt text][image8]
after thresholding:
![alt text][image9]
---

Video Implementation
---

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also saved the hot boxes of the last 12 frames, and applied the threshold to the sum of these last 12 hot box frames. This helped reduce false positives and made the bounding boxes more robust to misdetections. 


Discussion
---



1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main drawback of this approach is its speed. When running the pipeline on video it computed about 1 frame per second, which is nowhere close to real-time. One option for optimization would be to parallelize the feature extraction for different regions, and parallelize the sliding windows classification.
