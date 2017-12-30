import cv2
import numpy as np
import matplotlib.pyplot as plt

import util

cam_cal = np.load('./cam_cal.npz')


def _detect_yellow():
    # Detect color yellow
    offset = 180
    yellow_selector = (rgb_img[:, :, 0] >
                       offset) & (rgb_img[:, :, 1] > offset) & (
                           np.abs(rgb_img[:, :, 0] - rgb_img[:, :, 1]) < 80)
    yellow = np.copy(rgb_img)
    yellow[:, :, :] = 0
    yellow[yellow_selector] = [255, 255, 255]


def binary_thres(img, lower_pct=97, upper_pct=100, lower=None, upper=None):
    assert len(img.shape) == 2 or img.shape[0] == 1

    if lower is None:
        lower = np.percentile(img, lower_pct)
    if upper is None:
        upper = np.percentile(img, upper_pct)

    # print(lower, upper)

    binary = np.zeros_like(img)
    binary[(img >= lower) & (img <= upper)] = 1

    return binary


def sobel_thres(img, sobel_kernel=3):
    assert len(img.shape) == 2 or img.shape[-1] == 1

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    gradx = binary_thres(sobelx, lower=0, upper=255)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    grady = binary_thres(sobely, lower=0, upper=255)

    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    mag_bin = binary_thres(gradmag, lower=190, upper=255)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_margin = np.pi / 10
    lower_dir = np.pi / 2 - dir_margin
    upper_dir = np.pi / 2 + dir_margin
    condition = (absgraddir >= lower_dir) & (absgraddir < upper_dir)

    dir_bin = np.zeros_like(img)
    dir_bin[condition] = 1

    combined_bin = np.zeros_like(img)
    condition = (mag_bin == 1) | (dir_bin == 1) | (img == 1)
    combined_bin[condition] = 255

    return np.dstack([combined_bin] * 3)


def edge_detection(rgb_img, s_only=False, clahe=True):
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    else:
        clahe = None

    R = rgb_img[:, :, 0]
    if clahe:
        R = clahe.apply(R)

    r_binary = binary_thres(R)
    r_edge = sobel_thres(r_binary)  # helper.canny(img)

    hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]

    if clahe:
        S = clahe.apply(S)

    s_binary = binary_thres(S)
    s_edge = sobel_thres(s_binary)

    final = s_edge

    if not s_only:
        final = final // 2 + r_edge // 2

    return final


def undistort_img(img):
    mtx = cam_cal['mtx']
    dist = cam_cal['dist']

    return cv2.undistort(img, mtx, dist, None, None)


def bird_eye_view(img, src_corners, w, h, offset=10):
    dst_corners = np.array([(offset, offset), (w - offset, offset),
                            (w - offset, h - offset),
                            (offset, h - offset)]).astype('float32')
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    inv_M = cv2.getPerspectiveTransform(dst_corners, src_corners)
    dst = cv2.warpPerspective(img, M, (w, h))

    return dst, M, inv_M


def _detect_lane_px_from_fit(bin_img,
                             left_fit,
                             right_fit,
                             margin=200,
                             debug_lv=0):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "bin_img")
    # It's now much easier to find line pixels!
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = (
        (nonzerox >
         (left_fit[0] *
          (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
        (nonzerox <
         (left_fit[0] *
          (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_inds = (
        (nonzerox >
         (right_fit[0] *
          (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
        (nonzerox <
         (right_fit[0] *
          (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    out_img = None
    if debug_lv >= 1:
        # Create an image to draw on and an image to show the selection window
        out_img = np.ones_like(bin_img) * 255
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [
            255, 0, 0
        ]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
            0, 0, 255
        ]

        plt.imshow(out_img)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return (leftx, lefty, rightx, righty, out_img)


def _detect_lane_from_fit(bin_img, left_fit, right_fit, margin=100,
                          debug_lv=0):
    leftx, lefty, rightx, righty, out_img = _detect_lane_px_from_fit(
        bin_img, left_fit, right_fit, debug_lv=debug_lv)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if debug_lv >= 1:
        # Generate x and y values for plotting
        ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.title('from fit')

        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return (left_fit, right_fit, leftx, lefty, rightx, righty)


def _clean_lane_segment(win_img, min_px=800, connectivity=4, debug_lv=0):
    # Perform the operation
    labels, stats = cv2.connectedComponentsWithStats(win_img, connectivity,
                                                     cv2.CV_32S)[1:3]
    # selected_height = util.reject_outliers(stats[1:, cv2.CC_STAT_HEIGHT])
    areas = stats[1:, cv2.CC_STAT_AREA]
    area_selector = util.reject_outliers2(areas, min_val=min_px)
    selected_area = stats[1:, cv2.CC_STAT_AREA][area_selector]

    if len(selected_area) == len(stats[1:, cv2.CC_STAT_AREA]):
        return win_img.nonzero()[:2]

    if debug_lv >= 2:
        plt.figure()
        plt.imshow(labels)
        plt.show()

    selected_labels = 1 + area_selector[0]

    for selected in selected_labels:
        labels[labels == selected] = 255

    labels[labels != 255] = 0
    return labels.nonzero()[:2]


def _detect_lane_from_img(bin_img, margin=200, minpix=90, debug_lv=0):
    out_img = np.copy(bin_img)

    histogram = np.sum(bin_img[bin_img.shape[0] // 2:, :, 0], axis=0)

    if debug_lv >= 1:
        plt.figure()
        plt.plot(histogram)
        plt.imshow(out_img)
        plt.show()

    midpoint = np.int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    nwin = 15
    win_height = np.int(bin_img.shape[0] / nwin)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    # nonzero are in (h, w, chan)
    nonzeroy, nonzerox, _ = nonzero

    left_current = left_base
    left_last = left_base
    right_current = right_base
    right_last = right_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwin):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = bin_img.shape[0] - (window + 1) * win_height
        win_y_high = bin_img.shape[0] - window * win_height

        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # print(rect_area)
        # print(nonzerox.shape)
        # print(good_left_inds.shape)
        # print(len(good_left_inds), len(good_right_inds))
        # print('==')
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            selected = util.reject_outliers(nonzerox[good_left_inds])
            left_last = left_current
            left_current = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            # take the difference between last and current estimate and continue
            # to move in the same direction
            leftx_diff = left_current - left_last
            if leftx_diff > 0:
                # means we moved right, continue to move right
                left_last = left_current
                left_current -= leftx_diff // 2
            else:
                left_last = left_current
                left_current += leftx_diff // 2

        if len(good_right_inds) > minpix:
            selected = util.reject_outliers2(nonzerox[good_right_inds])

            if debug_lv >= 3:
                out_img[nonzeroy[good_right_inds[selected]], nonzerox[
                    good_right_inds[selected]]] = [255, 0, 0]
                plt.imshow(out_img)
                plt.show()

            right_last = right_current
            right_current = np.int(
                np.mean(nonzerox[good_right_inds[selected]]))
        else:
            # take the difference between last and current estimate and continue
            # to move in the same direction
            rightx_diff = right_current - right_last
            if rightx_diff > 0:
                # means we moved right, continue to move right
                right_last = right_current
                right_current -= rightx_diff // 2
            else:
                right_last = right_current
                right_current += rightx_diff // 2

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    left_lane_area = np.copy(bin_img[:, :, 0])
    left_lane_area[lefty, leftx] = 255
    left_lane_area[left_lane_area != 255] = 0
    ly, lx = _clean_lane_segment(left_lane_area, debug_lv=debug_lv)

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    right_lane_area = np.copy(bin_img[:, :, 0])
    right_lane_area[righty, rightx] = 255
    right_lane_area[right_lane_area != 255] = 0
    ry, rx = _clean_lane_segment(right_lane_area, debug_lv=debug_lv)

    if debug_lv >= 2:
        viz = np.copy(out_img[:, :, 0])
        viz[:, :] = 0
        viz[ry, rx] = 255
        plt.figure()
        plt.title("HEY")
        plt.imshow(viz)
        plt.show()

    # Fit a second order polynomial to each
    left_fit = np.polyfit(ly, lx, 2)
    right_fit = np.polyfit(ry, rx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    out_img[ly, lx] = [255, 0, 0]
    out_img[ry, rx] = [0, 0, 255]

    if debug_lv >= 2:
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return (left_fit, right_fit, leftx, lefty, rightx, righty)


def detect_lane(bin_img,
                left_lane,
                right_lane,
                margin=100,
                minpix=90,
                debug_lv=0):

    if left_lane.detected and right_lane.detected:
        left_fit = left_lane.best_fit
        right_fit = right_lane.best_fit

        (left_fit, right_fit, leftx, lefty, rightx,
         righty) = _detect_lane_from_fit(
             bin_img,
             left_fit,
             right_fit,
             margin=margin / 2,
             debug_lv=debug_lv)

        left_lane.update(left_fit, leftx, lefty)
        right_lane.update(right_fit, rightx, righty)

    if not left_lane.detected or not right_lane.detected:
        (left_fit, right_fit, leftx, lefty, rightx,
         righty) = _detect_lane_from_img(
             bin_img, margin, minpix, debug_lv=debug_lv)

        left_lane.update(left_fit, leftx, lefty)
        right_lane.update(right_fit, rightx, righty)

    return left_lane, right_lane


def draw_lanes(orig_img,
               bin_img,
               inv_M,
               left_lane,
               right_lane,
               w,
               h,
               margin=200,
               debug_lv=0):

    left_fit = left_lane.best_fit
    # print('left_fit', left_fit)
    right_fit = right_lane.best_fit
    # print('right_fit', right_fit)

    leftx, lefty, rightx, righty, out_img = _detect_lane_px_from_fit(
        bin_img,
        left_lane.best_fit,
        right_lane.best_fit,
        margin=margin,
        debug_lv=debug_lv)

    ploty = np.linspace(0, bin_img.shape[0] - 1, bin_img.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(bin_img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([
        np.transpose(np.vstack([left_fitx, ploty])),
    ])
    pts_right = np.array([
        np.flipud(np.transpose(np.vstack([right_fitx, ploty]))),
    ])
    pts = np.hstack((pts_left, pts_right))

    #print('HEY??', pts.shape)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # unwarp
    #print('HI::', color_warp.shape)
    newwarp = cv2.warpPerspective(color_warp, inv_M, (w, h))

    # Combine the result with the original image
    result = cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)

    if debug_lv >= 1:
        plt.imshow(result)
        plt.show()

    #print('Hi::', result)
    return result
