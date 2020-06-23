# TODO !!! not tested !!!
# -*- coding: utf-8 -*-

import argparse
import os
import time

import craft_text_detector
import craft_text_detector.file_utils as file_utils
import craft_text_detector.imgproc as imgproc
import cv2
from craft_text_detector.craft_detector_util import str2bool

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--trained_model', default='../craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--refiner_model', default='../craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--is_poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')

args = parser.parse_args()

# TODO! complete test that will test on entire dataset. I don't have data for now.
if __name__ == '__main__':

    test_folder = args.test_folder

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(test_folder)

    output_dir = './result/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # load net
    craft_model_path = args.trained_model

    refine = args.refine
    if refine:
        refinenet_model_path = args.refiner_model
    else:
        refinenet_model_path = None
    text_threshold = args.text_threshold
    low_text = args.low_text
    link_threshold = args.link_threshold
    cuda = args.cuda
    square_size = args.canvas_size
    mag_ratio = args.mag_ratio
    poly = args.poly
    show_time = args.show_time

    craft_net = craft_text_detector.craft_detector.craft_detector(craft_model_path=craft_model_path,
                                                                  refinenet_model_path=refinenet_model_path,
                                                                  cuda=cuda)  # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')

    args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        # image = imgproc.loadImage(image_path)
        image = imgproc.read_image(,

        # bboxes, polys, score_text = test_net(craft_net, image, args.text_threshold, args.link_threshold, args.low_text,
        #                                      args.cuda, args.is_poly, refine_net)

        # return {
        #     "boxes": boxes,
        #     "boxes_as_ratios": boxes_as_ratio,
        #     "polys": polys,
        #     "polys_as_ratios": polys_as_ratio,
        #     "heatmaps": {
        #         "text_score_heatmap": text_score_heatmap,
        #         "link_score_heatmap": link_score_heatmap,
        #     },
        #     "times": times,
        # }

        prediction = craft_net.get_prediction(image=image, text_threshold=text_threshold, link_threshold=link_threshold,
                                              low_text=low_text, square_size=square_size, mag_ratio=mag_ratio,
                                              poly=poly, show_time=show_time)
        bboxes, polys, score_text = prediction["boxes"], prediction["polys"], prediction["text_score_heatmap"]

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = output_dir + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.export_extra_results(image_path, image[:, :, ::-1], polys, output_dir=output_dir)

    print("elapsed time : {}s".format(time.time() - t))
