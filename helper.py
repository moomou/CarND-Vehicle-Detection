import cv2
import numpy as np

import util


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, sigma=0.33):
    """Applies the Canny transform
    Ref: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    """
    m = np.median(img)
    lower = int(max(0, (1.0 - sigma) * m))
    upper = int(min(255, (1.0 + sigma) * m))
    return cv2.Canny(img, lower, upper)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255, ) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def merge_lines(lines, h, state=None):
    if len(lines):
        line_pts = np.array(util.line2pts(lines))
        line_param = cv2.fitLine(line_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    else:
        line_param = None

    return util.extrapolate_line(line_param, h * 0.6, h, state=state)


def line2segs(lines):
    slopes = [util.get_slope(*line) for line in lines]

    left_slopes = []
    right_slopes = []
    left_idx = []
    right_idx = []
    for idx, slope in enumerate(slopes):
        # skip near horizontal lines and overly steep lines
        if abs(slope) < 0.3 or abs(slope) > 2:
            continue

        if slope > 0:
            right_idx.append(idx)
            right_slopes.append(slope)
        else:
            left_idx.append(idx)
            left_slopes.append(slope)

    left_slopes = np.array(left_slopes)
    right_slopes = np.array(right_slopes)

    left_lines = [
        lines[left_idx[idx]]
        for idx, valid in enumerate(util.reject_outliers(left_slopes)) if valid
    ]
    right_lines = [
        lines[right_idx[idx]]
        for idx, valid in enumerate(util.reject_outliers(right_slopes))
        if valid
    ]

    return left_lines, right_lines


def draw_lines2(img, lines, color=[255, 0, 0], thickness=20, state=None):
    h, w, chan = img.shape

    lines = [line[0] for line in lines]
    left_lines, right_lines = line2segs(lines)

    # print('left line')
    left_line, left_state = merge_lines(left_lines, h, state and
                                        (state.get('left') or {}))

    # print('right line')
    right_line, right_state = merge_lines(right_lines, h, state and
                                          (state.get('right') or {}))

    if state:
        state['left'] = left_state
        state['right'] = right_state

    for line in [left_line, right_line]:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=20):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, state):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines2(line_img, lines, state=state)

    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, a=0.8, beta=1., lam=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, beta, lam)
