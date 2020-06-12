import math

import cv2
import numpy as np
from skimage.measure import label, regionprops


# %% get_detection_boxes_core


def get_detection_boxes_core(textmap, linkmap, text_threshold: float = 0.7, link_threshold: float = 0.4,
                             low_text: float = 0.4, only_characters=False):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape
    # what is that shit? please take a look; https://en.wikipedia.org/wiki/Gaussian_function
    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, cv2.THRESH_BINARY)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, cv2.THRESH_BINARY)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    # text_score_comb = np.clip(text_score, 0, 1)
    # text_score_comb = np.clip(link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )
    if only_characters:
        labels = text_score

    detecteds = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = make_segmentation_map(k, labels, textmap)

        if not only_characters:
            # remove link area
            ex, ey, niter, sx, sy = remove_link_area(k, link_score, segmap, size, stats, text_score)

            # boundary check
            segmap = boundary_check(ex, ey, img_h, img_w, niter, segmap, sx, sy)

        # make box
        # TODO! extracts only one bb
        box, np_contours = make_box(segmap, only_characters)

        if not only_characters:
            # align diamond-shape
            box = align_diamond_shape(box, np_contours)

            # make clock-wise order
            box = make_clock_wise_order(box, only_characters)
        detecteds.append(box)
        mapper.append(k)

    # return detecteds, labels, mapper  # detected for each "nLabels"
    # assert not detecteds == [], "nothing to detect. image is too bad to make labeling"
    # assert not np.sum(labels) == 0, "image is too bad to make labeling"
    # assert not mapper == [], "image is too bad to make labeling"
    return detecteds, labels, mapper * detecteds[0].shape[0]  # detected for each "nLabels"


def make_clock_wise_order(box, only_characters=False):
    if only_characters:
        raise NotImplementedError
    else:
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
    return box


def make_segmentation_map(k, labels, textmap):
    segmap = np.zeros(textmap.shape, dtype=np.uint8)
    segmap[labels == k] = 255
    return segmap


def align_diamond_shape(box, np_contours):
    w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
    box_ratio = max(w, h) / (min(w, h) + 1e-5)
    if abs(1 - box_ratio) <= 0.1:
        l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
        t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
        box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
    return box


def make_box(segmap, only_characters=False):
    mask = segmap != 0
    notzero_tuple = np.where(mask)
    notzero = np.array(notzero_tuple)
    np_temp = np.roll(notzero, 1, axis=0)
    np_contours = np_temp.transpose().reshape(-1, 2)
    # TODO! extracts only one bb
    if not only_characters:
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        # array([[7., 14.],  # left, bottom
        #        [7., 4.],   # left, upper
        #        [70., 4.],  # right, upper
        #        [70., 14.]],# right, bottom
        #        dtype=float32)
    else:
        box = []
        lbl_0 = label(mask)
        props = regionprops(lbl_0)
        for prop in props:
            box.append(prop.bbox)
        # make it compatible
        box = np.array(box)
        compatible_box = np.zeros((box.shape[0], 4, 2), dtype=np.float32)

        # Bounding boxes expanded not to loss segmented area.
        #                                   y       ,   x
        box_coord_left_bottom = np.array([box[:, 1] - 1, box[:, 2]]).transpose()
        compatible_box[:, 0, :] = box_coord_left_bottom

        box_coord_left_upper = np.flip(box[:, 0:2], axis=1) - 1
        compatible_box[:, 1, :] = box_coord_left_upper
        #                                   y       ,   x
        box_coord_right_upper = np.array([box[:, 3], box[:, 0] - 1]).transpose()
        compatible_box[:, 2, :] = box_coord_right_upper

        box_coord_right_bottom = np.flip(box[:, 2:], axis=1)
        compatible_box[:, 3, :] = box_coord_right_bottom

        # TODO! sort the bounding boxes
        compatible_box = sort_bounindg_boxes(compatible_box, segmap, margin_y=3)
        box = compatible_box
    return box, np_contours


def sort_bounindg_boxes(box, segmap, margin_y=3):
    # %% only to detect multiple characters in once. each bounding box has 4 coordinate points.
    # more numpy, vectoring way.
    # left_upper = box[:, 1, :]

    left_upper_y = box[:, 1, 1]
    shorted_ids_y = np.argsort(left_upper_y)
    box = box[shorted_ids_y, :, :]

    # %% find how much margin between bounding boxes along x axis.
    # if it not bigger than margin_y they are in the same line.
    # box_margin_y = box[:, 1, 1][1:] - box[:, 1, 1][:-1]
    # box_margin_y = np.concatenate(([1], box_margin_y))
    #
    # mask_y = box_margin_y <= margin_y  # mask not bigger than margin_y
    # box = [box[mask_y, :, :]]  # select not bigger than margin_y
    # while not np.all(mask_y) or mask_y != None:  # Python don't have do-while loop
    #     box_margin_y = box[:, 1, 1][1:] - box[:, 1, 1][:-1]
    #     box_margin_y = np.concatenate(([1], box_margin_y))
    #
    #     mask_y = box_margin_y > margin_y  # mask bigger than margin_y
    #     box_same_line.append(box[mask_y, :, :])  # select not bigger than margin_y

    # TODO! !!!POSSIBLE BUG!!! assumed all in the same line.
    left_upper_x = box[:, 1, 0]
    shorted_ids_x = np.argsort(left_upper_x)
    box = box[shorted_ids_x, :, :]
    return box


    # another idea. conventional way.
    # lines = []
    # for p, bb_pivot in enumerate(box):
    #     pivot_point = bb_pivot[1]  # left_upper
    #
    #     # determinte [y, x] points and margin.
    #     pivot_point_y = pivot_point[0]
    #     pivot_point_x = pivot_point[1]
    #     # I am giving some margin to collect bb in the same line.
    #     # check for smaller than zero
    #     max_idx = segmap.shape[1]
    #     margin_top = np.clip(pivot_point_x - margin, 0, max_idx)
    #     margin_bottom = pivot_point_x + margin
    #
    #     same_line = [bb_pivot]
    #     box_reduced = box[np.arange(len(box)) != 3]
    #     for bb in box_reduced:
    #         pivot_point = bb_pivot[1]  # left_upper
    #
    #         # determinte [y, x] points and margin.
    #         pivot_point_y = pivot_point[0]
    #         pivot_point_x = pivot_point[1]


def remove_link_area(k, link_score, segmap, size, stats, text_score):
    segmap[np.logical_and(link_score == 1, text_score == 0)] = 0
    x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
    w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
    niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
    sx, ex, sy, ey = (x - niter, x + w + niter + 1, y - niter, y + h + niter + 1)
    return ex, ey, niter, sx, sy


def boundary_check(ex, ey, img_h, img_w, niter, segmap, sx, sy):
    if sx < 0:
        sx = 0
    if sy < 0:
        sy = 0
    if ex >= img_w:
        ex = img_w
    if ey >= img_h:
        ey = img_h
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
    segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
    return segmap
