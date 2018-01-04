import deco
import numpy as np
import cv2

from lesson_func import (convert_color, get_hog_features, bin_spatial,
                         color_hist)


def _find_car(patch, hog_features, svc, X_scaler, xleft, ystart, ytop, window,
              scale, spatial_size, hist_bins):
    # Get color features
    spatial_features = bin_spatial(patch, size=spatial_size)
    hist_features = color_hist(patch, nbins=hist_bins)

    # Scale features and make a prediction
    test_features = X_scaler.transform(
        np.hstack((spatial_features, hist_features, hog_features))
        .reshape(1, -1))

    test_prediction = svc.predict(test_features)

    if test_prediction == 1:
        xbox_left = np.int(xleft * scale)
        ytop_draw = np.int(ytop * scale)
        win_draw = np.int(window * scale)

        return (xbox_left, ytop_draw + ystart), (xbox_left + win_draw,
                                                 ytop_draw + win_draw + ystart)


# Define a single function that can extract features using hog sub-sampling and make predictions
@deco.concurrent
def find_cars(img,
              ystart,
              ystop,
              scale,
              svc,
              X_scaler,
              orient,
              pix_per_cell,
              cell_per_block,
              spatial_size,
              hist_bins,
              color_space,
              debug_lv=0):
    if type(spatial_size) != tuple:
        spatial_size = (spatial_size, spatial_size)

    img_ = img.astype(np.float32) / 255
    selector = slice(ystart, ystop)

    ctrans_tosearch = convert_color(
        img_[selector, :, :], conv='RGB2%s' % color_space)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        scale_param = (np.int(imshape[1] / scale), np.int(imshape[0] / scale))
        ctrans_tosearch = cv2.resize(ctrans_tosearch, scale_param)

    if debug_lv >= 2:
        cv2.imwrite('se_%s_%s_%s.png' % (ystart, ystop, scale),
                    ctrans_tosearch)

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 3  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(
        ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(
        ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(
        ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    pos_window = [None] * nxsteps * nysteps
    counter = 0
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(
                ctrans_tosearch[ytop:ytop + window, xleft:xleft + window],
                (64, 64))

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:
                             xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:
                             xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:
                             xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            pos_window[counter] = _find_car(
                subimg, hog_features, svc, X_scaler, xleft, ystart, ytop,
                window, scale, spatial_size, hist_bins)
            counter += 1

    return [win for win in pos_window if win is not None]
