# -*- coding: utf-8 -*-
import cv2
import numpy as np


def read_image(img_file):
    """
    Read image in numpy
    :param img_file: Input image
    :return: input ready image for numpy network (model)
    """
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # following two cases are not explained in the original repo
    if img.shape[0] == 2:
        img = img[0]
    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def normalize_mean_variance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    """
    Normalize mean and variance
    :param in_img: Input image
    :param mean: mean values
        Default: (0.485, 0.456, 0.406)
    :param variance: Variance values
        Default: (0.229, 0.224, 0.225)
    :return: normalized image
    """
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


def denormalize_mean_variance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    """
    Clip mean and variance
    :param in_img: Input image
    :param mean: mean values
        Default: (0.485, 0.456, 0.406)
    :param variance: Variance values
        Default: (0.229, 0.224, 0.225)
    :return: normalized image
    """
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, square_size, interpolation=cv2.INTER_CUBIC, mag_ratio=1):
    """
    Resize according to aspect radio.
    :param img: Input image
    :param square_size: target size of image
    :param interpolation: Interpolation type in opencv format
        Default: cv2.INTER_CUBIC
    :param mag_ratio: image magnification ratio
        Default: 1
    :return: (Resized image, target image ratio, heatmap size)
        (img_resized, target_ratio, size_heatmap)
    """
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def cvt2HeatmapImg(img):
    """
    Generates heatmap in numpy
    :param img: Input image
    :return: heatmap in numpy
    """
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
