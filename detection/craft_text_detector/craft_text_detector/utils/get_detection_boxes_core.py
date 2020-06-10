import math

import cv2
import numpy as np
from skimage.measure import label, regionprops

#%% get_detection_boxes_core


def get_detection_boxes_core(textmap, linkmap, text_threshold: float = 0.7, link_threshold: float = 0.4,
                             low_text: float = 0.4, only_characters=False):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

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
        box = make_clock_wise_order(box)
        detecteds.append(box)
        mapper.append(k)

    return detecteds, labels, mapper  # detected for each "nLabels"


def make_clock_wise_order(box):
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

        box_coord_left_bottom = np.array([box[:, 1], box[:, 2]]).transpose()
        compatible_box[:, 0, :] = box_coord_left_bottom

        box_coord_left_upper = np.flip(box[:, 0:2], axis=1)
        compatible_box[:, 1, :] = box_coord_left_upper

        box_coord_right_upper = np.array([box[:, 3], box[:, 0]]).transpose()
        compatible_box[:, 2, :] = box_coord_right_upper

        box_coord_right_bottom = np.flip(box[:, 2:], axis=1)
        compatible_box[:, 3, :] = box_coord_right_bottom

        box = compatible_box
    return box, np_contours


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