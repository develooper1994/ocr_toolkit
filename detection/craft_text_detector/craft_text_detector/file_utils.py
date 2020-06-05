# -*- coding: utf-8 -*-
import copy
import os

import cv2
import gdown
import numpy as np


def download(url: str, save_path: str):
    """
    Downloads file from my gdrive, shows progress.
        Example inputs:
            url: 'ftp://smartengines.com/midv-500/dataset/01_alb_id.zip'
            save_path: 'data/file.zip'
    :param url: Download url
    :type url: str
    :param save_path: saving directory
    :type save_path: str
    :return: None
    """

    # create save_dir if not present
    create_dir(os.path.dirname(save_path))
    # download file
    gdown.download(url, save_path, quiet=False)


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    :param _dir: Directory
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def get_files(img_dir):
    """
    Get all image files
    :param img_dir: Input image directory
    :return: (images files, masks files, xml metadata files)
    """
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(in_path):
    """
    Get and list all images in disk
    :param in_path: Input path that include images
    :return: (images files, masks files, xml metadata files)
    """
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if (
                ext == ".jpg"
                or ext == ".jpeg"
                or ext == ".gif"
                or ext == ".png"
                or ext == ".pgm"
            ):
                img_files.append(os.path.join(dirpath, file))
            elif ext == ".bmp":
                mask_files.append(os.path.join(dirpath, file))
            elif ext == ".xml" or ext == ".gt" or ext == ".txt":
                gt_files.append(os.path.join(dirpath, file))
            elif ext == ".zip":
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


def rectify_poly(img, poly):
    """
    Scene is not always looks directly to the writing. So need to reduce variation of looking angle and size.
    Rectifies image
    :param img: Input image
    :param poly: Predicted bounding boxes in poligonal shape
    :return: Rectified image
    """
    # Use Affine transform
    n = int(len(poly) / 2) - 1
    width = 0
    height = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        width += int(
            (np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2
        )
        height += np.linalg.norm(box[1] - box[2])
    width = int(width)
    height = int(height / n)

    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    width_step = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        w = int((np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2)

        # Top triangle
        pts1 = box[:3]
        pts2 = np.float32(
            [[width_step, 0], [width_step + w - 1, 0], [width_step + w - 1, height - 1]]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        # Bottom triangle
        pts1 = np.vstack((box[0], box[2:]))
        pts2 = np.float32(
            [
                [width_step, 0],
                [width_step + w - 1, height - 1],
                [width_step, height - 1],
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        cv2.line(
            warped_mask, (width_step, 0), (width_step + w - 1, height - 1), (0, 0, 0), 1
        )
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        width_step += w
    return output_img


def crop_poly(image, poly):
    """
    Crops poligon from image
    :param image: Input image
    :param poly: Predicted bounding boxes in poligonal shape
    :return: croped image
    """
    # points should have 1*x*2  shape
    if len(poly.shape) == 2:
        poly = np.array([np.array(poly).astype(np.int32)])

    # create mask with shape of image
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)

    # method 1 smooth region
    cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))

    # crop around poly
    res = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(poly)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

    return cropped


def export_poly(image, poly, rectify, gray_scale=False):
    """
    :param image: full image
    :param points: bbox or poly points
    :param file_path: path to be exported
    :param rectify: rectify detected polygon by affine transform
    :param gray_scale: If you want in grayscale set it True
        Default; False
    :return: region result
    """
    if rectify:
        # rectify poly region
        detection_result = rectify_poly(image, poly)
    else:
        detection_result = crop_poly(image, poly)

    if gray_scale:
        detection_result = cv2.cvtColor(detection_result, cv2.COLOR_BGR2GRAY)
    return detection_result


def export_detected_region(image, poly, file_path, rectify=True, is_save=False, gray_scale=False):
    """
    :param image: full image
    :param points: bbox or poly points
    :param file_path: path to be exported
    :param rectify: rectify detected polygon by affine transform
    :param is_save: Saving if it is "True"
    :param gray_scale: If you want in grayscale set it True
        Default; False
    :return: region result
    """
    result = export_poly(image, poly, rectify, gray_scale=gray_scale)

    # export corpped region
    if is_save:
        cv2.imwrite(file_path, result)

    return result


def export_detected_polygons(image, regions, rectify: bool = False, gray_scale=False):
    """
    Export regions.
    :param image: full/original image
    :param regions: list of bboxes or polys
    :param rectify: rectify detected polygon by affine transform
    :param gray_scale: If you want in grayscale set it True
        Default; False
    :return: detected polygons
    """
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # init exported results
    detection_results = []

    for ind, region in enumerate(regions):
        # export region
        detection_result = export_poly(image, poly=region, rectify=rectify, gray_scale=gray_scale)
        # note exported results
        detection_results.append(detection_result)

    return detection_results


def export_detected_regions(image_path, image, regions, output_dir: str = "output/", rectify: bool = False):
    """
    Export and save regions as image files.
    :param image_path: path to original image
    :param image: full/original image
    :param regions: list of bboxes or polys
    :param output_dir: folder to be exported
    :param rectify: rectify detected polygon by affine transform
    :return: exported file(image) paths
    """
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # init exported file paths
    exported_file_paths = []

    # get file name
    file_name, file_ext = os.path.splitext(os.path.basename(image_path))

    # create crops dir
    crops_dir = os.path.join(output_dir, file_name + "_crops")
    create_dir(crops_dir)

    # export regions
    for ind, region in enumerate(regions):
        # get export path
        file_path = os.path.join(crops_dir, "crop_" + str(ind) + ".png")
        # export region
        export_detected_region(image, poly=region, file_path=file_path, rectify=rectify, is_save=True)
        # note exported file path
        exported_file_paths.append(file_path)

    return exported_file_paths


def export_extra_results(
    image_path,
    image,
    regions,
    heatmaps,
    output_dir="output/",
    texts=None,
):
    """ save text detection result one by one
    Args:
        image_path (str): image file name
        image (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4]
            for QUAD output
    Return:
        None
    """
    image = np.array(image)

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(image_path))

    # result directory
    res_file = os.path.join(output_dir, filename + "_text_detection.txt")
    res_img_file = os.path.join(output_dir, filename + "_text_detection.png")
    text_heatmap_file = os.path.join(output_dir, filename + "_text_score_heatmap.png")
    link_heatmap_file = os.path.join(output_dir, filename + "_link_score_heatmap.png")

    # create output dir
    create_dir(output_dir)

    # export heatmaps
    cv2.imwrite(text_heatmap_file, heatmaps["text_score_heatmap"])
    cv2.imwrite(link_heatmap_file, heatmaps["link_score_heatmap"])

    with open(res_file, "w") as f:
        for i, region in enumerate(regions):
            region = np.array(region).astype(np.int32).reshape((-1))
            strResult = ",".join([str(r) for r in region]) + "\r\n"
            f.write(strResult)

            region = region.reshape(-1, 2)
            cv2.polylines(
                image,
                [region.reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    (region[0][0] + 1, region[0][1] + 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness=1,
                )
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    tuple(region[0]),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness=1,
                )

    # Save result image
    cv2.imwrite(res_img_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
