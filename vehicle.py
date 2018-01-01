#!/usr/bin/env python
import glob
import json
import os

import matplotlib.image as mpimg
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from scipy.stats import randint
from skimage.feature import hog
from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from lesson_func import (
    heatmap,
    extract_features,
    slide_window,
    search_windows,
    draw_boxes,
    add_heat, )
import util

### TODO: Tweak these parameters and see how the results change.
'''
color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 3  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
'''

rand_state = 42


@util.h5cache
def cached_extract_features(*args, **kwargs):
    return extract_features(*args, **kwargs)


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


class HogEstimator(BaseEstimator):
    def __init__(self,
                 color_space='LUV',
                 orient=9,
                 pix_per_cell=8,
                 cell_per_block=3,
                 hog_channel='ALL',
                 spatial_size=(16, 16),
                 hist_bins=16,
                 spatial_feat=True,
                 hist_feat=True,
                 hog_feat=True):
        # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.color_space = color_space
        # HOG orientations
        self.orient = orient
        # HOG pixels per cell
        self.pix_per_cell = pix_per_cell
        # HOG cells per block
        self.cell_per_block = cell_per_block
        # Can be 0, 1, 2, or "ALL"
        self.hog_channel = hog_channel
        # Spatial binning dimensions
        self.spatial_size = spatial_size
        # Number of histogram bins
        self.hist_bins = hist_bins
        # Spatial features on or off
        self.spatial_feat = spatial_feat
        # Histogram features on or off
        self.hist_feat = hist_feat
        # HOG features on or off
        self.hog_feat = hog_feat

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def transform(self, X, y=None):
        print('params:', self.color_space, self.spatial_size, self.hist_bins,
              self.orient, self.pix_per_cell, self.cell_per_block,
              self.hog_channel, self.spatial_feat, self.hist_feat,
              self.hog_feat)

        features = extract_features(
            X,
            color_space=self.color_space,
            spatial_size=self.spatial_size,
            hist_bins=self.hist_bins,
            orient=self.orient,
            pix_per_cell=self.pix_per_cell,
            cell_per_block=self.cell_per_block,
            hog_channel=self.hog_channel,
            spatial_feat=self.spatial_feat,
            hist_feat=self.hist_feat,
            hog_feat=self.hog_feat)

        return np.array(features).astype(np.float64)

    def fit(self, X, y=None):
        return self


def search(car_dir='./vehicles', notcar_dir='non-vehicles', debug_lv=0):
    imgs = get_test_imgs()
    h, w, c = imgs[0].shape

    cars = glob.glob(os.path.join(car_dir, '**/*.png'))
    notcars = glob.glob(os.path.join(notcar_dir, '**/*.png'))

    params = {
        'hog__color_space': [
            'RGB',
            'HSV',
            'LUV',
            'HLS',
            'YUV',
            'YCrCb',
        ],
        'hog__hist_bins': [
            2,
            4,
            8,
            12,
        ],
        'hog__orient': randint(low=4, high=12),
        'hog__pix_per_cell': randint(low=1, high=4),
        'hog__cell_per_block': randint(low=3, high=2**3),
        'hog__spatial_size': [
            8,
            16,
            32,
        ],
    }
    X_scaler = StandardScaler()
    hog_estimate = HogEstimator()
    pipe = Pipeline([('hog', hog_estimate), ('scaler', X_scaler),
                     ('svc', LinearSVC())])

    search = RandomizedSearchCV(
        pipe, param_distributions=params, n_iter=5, n_jobs=1, pre_dispatch=1)

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 250
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    X = np.array(cars + notcars)
    y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

    print(X.shape)
    print(y.shape)

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Check the training time for the SVC
    t = time.time()
    search.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    acc = round(search.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', acc)
    # Check the prediction time for a single sample
    t = time.time()

    cvres = search.cv_results_
    # for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    # print(np.sqrt(-mean_score), params)
    print(cvres.keys())

    print('Bestparam:', search.best_params_)
    with open('./hog_params.json', 'w') as f:
        f.write(json.dumps(search.best_params_))

    joblib.dump(search.best_estimator_, 'pipe.pickle')


def train(car_dir='./vehicles', notcar_dir='non-vehicles', debug_lv=0):
    with open('./hog_param.json') as f:
        hog_params = json.load(f.read())

    imgs = get_test_imgs()
    h, w, c = imgs[0].shape

    cars = glob.glob(os.path.join(car_dir, '**/*.png'))
    notcars = glob.glob(os.path.join(notcar_dir, '**/*.png'))

    X_scaler = StandardScaler()
    hog_estimate = HogEstimator(**hog_params)
    svc = LinearSVC()

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    if debug_lv >= 1:
        sample_size = 250
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    X = np.array(cars + notcars)
    X = hog_estimate.transform(X)
    X = X_scaler.fit_transform(X)
    y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

    print(X.shape)
    print(y.shape)

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    acc = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', acc)

    joblib.dump('./svc.pickle')
    joblib.dump('./xscaler.pickle')

    test(debug_lv=debug_lv)


def test(test_out_dir='./output_images', debug_lv=0):
    with open('./hog_param.json') as f:
        hog_params = json.load(f.read())

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

        hot_windows = search_windows(img, all_windows, svc, X_scaler,
                                     **hog_params)

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
        's': search,
        't': train,
        'test': test,
    })
