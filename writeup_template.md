## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car](asset/car.png)
![not car](asset/notcar.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using optimized parameters (see section below) for a car and noncar input.

Car
![hog sample](asset/car_hog.png)

Not car
![hog sample](asset/notcar_hog.png)

The code for HOG can be found in lesson_func.py `get_hog_features` and `single_img_features`.

#### 2. Explain how you settled on your final choice of HOG parameters.

To identify the best parameters for the HOG features, I utilized `RandomizedSearchCV` in scikit learn to search for best the best peforming parameters for over a few hundred images from car and non-car class.

After the experiment, I obtained the following parameters:
```
{"hog__hist_bins": 16, "hog__cell_per_block": 4, "hog__spatial_size": 8, "hog__orient": 10, "hog__pix_per_cell": 8, "hog__color_space": "HLS"}
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM as demonstrated in the lesson and the code can be found in vehicle.py.

For the detailed HOG parameters used, see the previous section. I was able to obtain test set accuracy of 98%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search can be found in vehicle.py `get_sliding_win`; this function utilizes the sliding window function from the lesson.

The search strategy is conducted using 64x64px windows. The actual region each window type sweep can be seen in the following image.

![sliding win](asset/window.png)

The scale and overlap ratio was best trial and error and the intuition that the classifier is trained on 64x64px patches so I used that as the minimum size box.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two frames with 2 different sliding windows overlapping rations, specifically 0.75 and 0.5. Additionally, one of the frame is blurred to remove potential noises. In terms of performance of optimization, I run the 2 separate search in parallel using python multiprocessing via `deco` library. Further optimization of finer grain parallelism is possible.

Example as applied to test images
![Ex1](asset/0.png)

![Ex2](asset/1.png)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Additionally, using the assumption that vehicles move relatively slowly in each frame, I retain heatmap values over long periods and decrement heatmap values slowly every frame divisible by 13. See code in pipe.py `vehicle_pipe`

Here's an example result showing the heatmap from a series of frames of video and before and after heatmap thresholding

### Here are some frames and their corresponding heatmaps:

Before #1
![Before](asset/before_detected_heatmap.png)

After #1
![After](asset/detected_heatmap.png)

Before #1
![Before](asset/before_detected_heatmap.png)

After #1
![After](asset/detected_heatmap.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

