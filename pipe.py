#!/usr/bin/env python
import os
import glob
from importlib import reload
from collections import defaultdict
from moviepy.editor import VideoFileClip

import cv2
import matplotlib.image as mpimg
import numpy as np

import helper
import helper2
import util
from line import Line, xm_per_pix

helper = reload(helper)
util = reload(util)

_state_cache = defaultdict(dict)

# Checkerboard pattern corners
NX = 9
NY = 6


def cam_calibration(viz=False):
    # Ref: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    objp = np.zeros((NX * NY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

    cam_imgs = [cv2.imread(f) for f in glob.glob('./camera_cal/*')]
    cam_imgs2 = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in cam_imgs]

    ret_corners = [
        cv2.findChessboardCorners(img, (NX, NY), None) for img in cam_imgs2
    ]

    objpoints = []
    imgpoints = []
    for idx, (ret, corners) in enumerate(ret_corners):
        if not ret:
            continue

        objpoints.append(objp)
        imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, cam_imgs2[0].shape[::-1], None, None)

    np.savez_compressed('cam_cal', mtx=mtx, dist=dist)

    if viz:
        idx = 0
        ret, corners = ret_corners[idx]
        img = cv2.drawChessboardCorners(cam_imgs[idx], (NX, NY), corners, ret)
        cv2.imwrite('sample.png', img)

        img = cv2.undistort(img, mtx, dist, None, None)
        cv2.imwrite('undist.png', img)

    return mtx, dist


def lane_pipe(rgb_img, state_id=None, debug_lv=0):
    '''
    The goals / steps of this project are the following:
        x Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        x Apply a distortion correction to raw images.
        x Use color transforms, gradients, etc., to create a thresholded binary image.
        x Apply a perspective transform to rectify binary image ("birds-eye view").
        x Detect lane pixels and fit to find the lane boundary.
        x Determine the curvature of the lane and vehicle position with respect to center.
        x Warp the detected lane boundaries back onto the original image.
        x Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    '''
    global _state_cache

    if state_id is None:
        # keep no state
        state = {}
        right_lane = Line()
        left_lane = Line()
    else:
        state = _state_cache[state_id]
        left_lane = state.get('left_lane') or Line()
        right_lane = state.get('right_lane') or Line()

    h, w, chan = rgb_img.shape
    region = np.array([
        # top left
        (int(w / 2) - 100, int(h / 2) + 100),
        # top right
        (int(w / 2) + 100, int(h / 2) + 100),
        # bottom right
        (int(w * 0.8), int(h * 0.85)),
        # bottom left
        (int(w * 0.2), int(h * 0.85)),
    ])

    img = helper2.undistort_img(rgb_img)
    img = helper.gaussian_blur(img, 5)
    img = helper2.edge_detection(img)

    # cv2.imwrite('edge_detection.png', img)

    img = helper.region_of_interest(img, [region])

    # cv2.imwrite('region.png', img)
    img, M, inv_M = helper2.bird_eye_view(img, region.astype('float32'), w, h)

    left_lane, right_lane = helper2.detect_lane(
        img, left_lane, right_lane, debug_lv=debug_lv)

    img = helper2.draw_lanes(rgb_img, img, inv_M, left_lane, right_lane, w, h)
    cv2.imwrite('unwarped.png', img)

    curavture_rad = left_lane.curvature(h) + right_lane.curvature(h)
    center_offset = np.abs(w / 2 -
                           (left_lane.base_x(h) + right_lane.base_x(h)) / 2)

    txt1 = 'Est. curv = %.2fm' % curavture_rad
    txt2 = 'Est. center offset = %.2fm' % (center_offset * xm_per_pix)
    cv2.putText(img, txt1, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.putText(img, txt2, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    state['left_lane'] = left_lane
    state['right_lane'] = right_lane

    return img


def process_image(output_root='./output_images',
                  img_root='test_images',
                  debug_lv=0):
    test_imgs = os.listdir(img_root)

    for path in test_imgs:
        if not path.endswith('5.jpg'):
            continue

        img = mpimg.imread(os.path.join(img_root, path))
        img = lane_pipe(img, debug_lv=debug_lv)

        cv2.imwrite(
            os.path.join(output_root, path),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def process_video(vidpath, start=None, end=None, debug_lv=0):
    state_id = os.path.basename(vidpath)
    output_path = os.path.join('./test_videos_output', state_id)

    def process_image(image):
        result = lane_pipe(image, state_id, debug_lv)
        return result

    clip = VideoFileClip(vidpath)

    if start and end:
        clip = clip.subclip(float(start), float(end))

    out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(output_path, audio=False)


if __name__ == '__main__':
    import fire

    fire.Fire({
        'proc': process_image,
        'cam': cam_calibration,
        'vid': process_video,
    })
