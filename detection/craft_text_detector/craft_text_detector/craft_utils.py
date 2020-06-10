# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np
import torch
from torch.backends import cudnn

""" auxilary functions """


# unwarp corodinates


def warp_coord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


""" end of auxilary functions """


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

    det = []
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
        box, np_contours = make_box(segmap)

        # align diamond-shape
        box = align_diamond_shape(box, np_contours)

        # make clock-wise order
        box = make_clock_wise_order(box)
        det.append(box)
        mapper.append(k)

    return det, labels, mapper


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


def make_box(segmap):
    np_temp = np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
    np_contours = np_temp.transpose().reshape(-1, 2)
    rectangle = cv2.minAreaRect(np_contours)
    box = cv2.boxPoints(rectangle)
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


def get_poly_core(boxes, labels, mapper):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = (
            int(np.linalg.norm(box[0] - box[1]) + 1),
            int(np.linalg.norm(box[1] - box[2]) + 1),
        )
        if w < 10 or h < 10:
            polys.append(None)
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None)
            continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:, i] != 0)[0]
            if len(region) < 2:
                continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len:
                max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None)
            continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg  # segment width
        pp = [None] * num_cp  # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0, len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0:
                    break
                cp_section[seg_num] = [
                    cp_section[seg_num][0] / num_sec,
                    cp_section[seg_num][1] / num_sec,
                ]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [
                cp_section[seg_num][0] + x,
                cp_section[seg_num][1] + cy,
            ]
            num_sec += 1

            if seg_num % 2 == 0:
                continue  # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1) / 2)] = (x, cy)
                seg_height[int((seg_num - 1) / 2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh
        # is smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None)
            continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:  # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = -math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (
                pp[2][1] - pp[1][1]
        ) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (
                pp[-3][1] - pp[-2][1]
        ) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if (
                        np.sum(np.logical_and(word_label, line_img)) == 0
                        or r + 2 * step_r >= max_r
                ):
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if (
                        np.sum(np.logical_and(word_label, line_img)) == 0
                        or r + 2 * step_r >= max_r
                ):
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None)
            continue

        # make final polygon
        poly = [warp_coord(Minv, (spp[0], spp[1]))]
        for p in new_pp:
            poly.append(warp_coord(Minv, (p[0], p[1])))
        poly.append(warp_coord(Minv, (epp[0], epp[1])))
        poly.append(warp_coord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warp_coord(Minv, (p[2], p[3])))
        poly.append(warp_coord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def get_detection_boxes(textmap, linkmap, text_threshold: float = 0.7, link_threshold: float = 0.4,
                        low_text: float = 0.4,
                        poly=False,
                        only_characters=False):
    boxes, labels, mapper = get_detection_boxes_core(
        textmap, linkmap, text_threshold, link_threshold, low_text, only_characters
    )

    if poly:
        polys = get_poly_core(boxes, labels, mapper)
    else:
        polys = [None] * len(boxes)

    return boxes, polys


def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def device_selection_helper_pytorch(device):
    is_device = torch.cuda.is_available()
    if device is None or device == 'gpu':
        assert is_device, "!!!CUDA is not available!!!"  # Double check ;)
        device_object = torch.device('cuda')
        cudnn.enabled = True
        cudnn.benchmark = False
    # elif device == 'auto':
    #     if num_device == 1:
    #         device_object = mx.gpu(num_device - 1) if mx.context.num_gpus() > 0 else mx.cpu(num_device - 1)
    #     else:
    #         device_object = [mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu() for i in range(num_device)]
    #         # device = [mx.gpu(i) for i in range(num_device)] if mx.context.num_gpus() > 0 else [mx.cpu(i) for i in
    #         #                                                                                    range(num_device)]
    elif device == 'cpu':
        device_object = torch.device('cpu')
    elif device == 'auto':
        device_object = None  # default for Pytorch. Use all.
    else:
        # If it isn't a string.
        print("Assuming device is a mxnet ctx or device object or queue. Exp: mx.gpu(0)")
        device_object = device
    return device_object
