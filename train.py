#!/usr/bin/env python
import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from lesson_func import (
    heatmap,
    extract_features,
    slide_window,
    search_windows,
    draw_boxes, )

### TODO: Tweak these parameters and see how the results change.
color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 3  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off


def get_test_imgs(test_dir='./test_images'):
    imgs = [
        mpimg.imread(g) for g in glob.glob(os.path.join(test_dir, '*.jpg'))
    ]
    # RGB, H, W, C
    return imgs


def get_sliding_win(w, h):
    # divide h into 3 sections
    base_h = h // 2

    # small win
    sm_windows = slide_window(
        x_start_stop=[0, w],
        y_start_stop=[base_h, base_h + int(base_h * 0.1)],
        xy_window=(32, 32),
        xy_overlap=(0.5, 0.5))

    # med win
    md_windows = slide_window(
        x_start_stop=[0, w],
        y_start_stop=[
            base_h + int(base_h * 0.1) // 2, base_h + int(base_h * 0.5)
        ],
        xy_window=(64, 64),
        xy_overlap=(0.5, 0.5))

    lg_windows = slide_window(
        x_start_stop=[0, w],
        y_start_stop=[base_h + int(base_h * 0.5) // 2, h],
        xy_window=(96, 96),
        xy_overlap=(0.5, 0.5))

    return sm_windows + md_windows + lg_windows


def train(car_dir='./vehicles', notcar_dir='non-vehicles', debug_lv=0):
    imgs = get_test_imgs()
    h, w, c = imgs[0].shape

    cars = glob.glob(os.path.join(car_dir, '**/*.png'))
    notcars = glob.glob(os.path.join(notcar_dir, '**/*.png'))

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    if debug_lv >= 1:
        sample_size = 500
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    car_features = extract_features(
        cars,
        color_space=color_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        hog_channel=hog_channel,
        spatial_feat=spatial_feat,
        hist_feat=hist_feat,
        hog_feat=hog_feat)

    notcar_features = extract_features(
        notcars,
        color_space=color_space,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        hog_channel=hog_channel,
        spatial_feat=spatial_feat,
        hist_feat=hist_feat,
        hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    print('X', X.shape)
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    print('sX', scaled_X.shape)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    joblib.dump(svc, 'svc.pickle')
    joblib.dump(X_scaler, 'xscaler.pickle')

    test(debug_lv=debug_lv)


def test(test_out_dir='./output_images', debug_lv=0):
    imgs = get_test_imgs()
    h, w, c = imgs[0].shape

    all_windows = get_sliding_win(w, h)

    svc = joblib.load('./svc.pickle')
    X_scaler = joblib.load('./xscaler.pickle')

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    for idx, orig_img in enumerate(imgs):
        img = orig_img.astype(np.float32) / 255

        hot_windows = search_windows(
            img,
            all_windows,
            svc,
            X_scaler,
            color_space=color_space,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            hog_channel=hog_channel,
            spatial_feat=spatial_feat,
            hist_feat=hist_feat,
            hog_feat=hog_feat)

        window_img = heatmap(orig_img, hot_windows, debug_lv=debug_lv)
        cv2.imwrite(os.path.join(test_out_dir, '%s.png' % idx), window_img)

        if debug_lv >= 1:
            window_img = draw_boxes(
                np.copy(img), hot_windows, color=(0, 0, 255), thick=6)

            plt.imshow(window_img)
            plt.show()


if __name__ == '__main__':
    import fire
    fire.Fire({
        't': train,
        'test': test,
    })
