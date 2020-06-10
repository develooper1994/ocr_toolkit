import time
from logging import warning
from pprint import pprint

import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch

# from recognition.handwritten_text_recognition.recognition.utils import device_selection_helper_pytorch
from detection.craft_text_detector.craft_text_detector.craft_utils import device_selection_helper_pytorch

try:
    # direct call
    from craft_text_detector import craft_utils
    from craft_text_detector import imgproc
    # my google drive
    from craft_text_detector.craft_detector_util import copyStateDict, get_weight_path
    from craft_text_detector.file_utils import (
        export_detected_regions,
        export_extra_results
    )
    from craft_text_detector.imgproc import read_image
    from craft_text_detector.models.craftnet import CRAFT
    from craft_text_detector.models.refinenet import RefineNet
except:
    # indirect call
    try:
        import craft_utils
        import imgproc
        # my google drive
        from craft_detector_util import copyStateDict, get_weight_path
        from file_utils import (
            export_detected_regions,
            export_extra_results
        )
        from imgproc import read_image
        from models.craftnet import CRAFT
        from models.refinenet import RefineNet
    except:
        from . import craft_utils
        from . import imgproc
        # my google drive
        from .craft_detector_util import copyStateDict, get_weight_path
        from .file_utils import (
            export_detected_regions,
            export_extra_results, export_detected_polygons
        )
        from .imgproc import read_image
        from .models.craftnet import CRAFT
        from .models.refinenet import RefineNet

# from . import (
#     craft_utils,
#     read_image,
#     export_detected_regions,
#     export_extra_results,
# )

# Original
# CRAFT_GENERAL_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
# CRAFT_IC15_GDRIVE_URL = "https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf"
# REFINENET_GDRIVE_URL = "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"

# My Google Drive connections. No difference from the original.
CRAFT_GENERAL_GDRIVE_URL = "https://drive.google.com/open?id=1CV3Ao4zuDikOkuHHNCw0_YRVWq7YsPBQ"
CRAFT_IC15_GDRIVE_URL = "https://drive.google.com/open?id=1zQYaWF9_9Jsu5xjA5tD0X9N6Ug0lnbtm"
REFINENET_GDRIVE_URL = "https://drive.google.com/open?id=1ZDe0WRwlxLRkwofQt8C18ZTF-oA3vbfs"


# !!! New oops way.
# TODO! Detected articles will be sorted according to the coordinate.
class craft_detector:
    """
    Craft(Character Region Awareness for Text Detection) implementation
    """

    def __init__(self, image=None,
                 craft_model_path=None,
                 refinenet_model_path=None,
                 crop_type="poly",
                 device: str = "cpu",
                 benchmark: bool = False):
        """
        Configures class initializer.
        :param image: Input image
        :param craft_model_path: craft network(model) path with name
        :param refinenet_model_path: refiner network(model) path with name
        :param device: device switch. "gpu", "cpu", "auto"
        :param benchmark: cudnn benchmark mode switch
        :return: None
        """
        self.predicted_polygon_image = 0
        self.detection_result = 0
        self.reload(image=image, craft_model_path=craft_model_path, refinenet_model_path=refinenet_model_path,
                    crop_type=crop_type, device=device, benchmark=benchmark)

    def __call__(self, *args, **kwargs):
        return self.get_prediction(**kwargs)

    def reload(self, image=None,
               craft_model_path=None,
               refinenet_model_path=None,
               crop_type="poly",
               device: str = "cpu",
               benchmark: bool = False):
        """
        Configures class initializer. Sometimes the class needs to be reloaded (configured) with new parameters
        :param image: Input image
        :param craft_model_path: craft network(model) path with name
        :param refinenet_model_path: refiner network(model) path with name
        :param device: device switch. "gpu", "cpu", "auto"
        :param benchmark: cudnn benchmark mode switch
        :return: None
        """
        # load craft input image
        self.image = self.set_image(image)
        # load craft net and refine net
        self.craft_net = CRAFT()  # initialize
        self.refine_net = RefineNet()  # initialize
        # Double check device
        self.__set_device(device, benchmark)

        # crop type
        self.crop_type = crop_type

        # models my google drive
        self.__set_craft_net(craft_model_path)

        self.__set_refine_net(refinenet_model_path)

    def set_image(self, image):
        """
        Configure input image. if image is string then tries to access path.
        :param image: input image or input image path
        :return: input image
        """
        self.image = image  # consider image is numpy-array or some tensor
        if isinstance(image, str):
            # consider image is path of image
            self.image = read_image(image)  # numpy image
        return self.image

    def __set_device(self, device: str = "cpu", benchmark: bool = False):
        """
        Detects device(CPU/GPU) and configures device that network(model) running on.
        :param device: device switch. "gpu", "cpu", "auto"
            Default: False
        :param benchmark: cudnn benchmark mode switch.
            Default: False
        :return: None
        """
        # self.cuda = cuda
        # self.benchmark = benchmark
        # self.is_device = torch.cuda.is_available()
        # # self.device = torch.device('cuda' if self.is_device and self.cuda else 'cpu')
        # if self.cuda and self.is_device:
        #     assert self.is_device, "!!!CUDA is not available!!!"  # Double check ;)
        #     self.device = torch.device('cuda')
        #     cudnn.enabled = True
        #     cudnn.benchmark = self.benchmark
        # else:
        #     self.device = torch.device('cpu')
        self.device = device_selection_helper_pytorch(device=device)

    def __set_craft_net(self, craft_model_path, model_switch=1):
        self.CRAFT_SYNDATA_GDRIVE_URL = "https://drive.google.com/open?id=1pzPBZ5cYDCHPVRYbWTgIjhntA_LLSLyS"
        self.craft_SYNDATA_model_name = "Syndata.pth"
        self.CRAFT_IC15_GDRIVE_URL = "https://drive.google.com/open?id=1zQYaWF9_9Jsu5xjA5tD0X9N6Ug0lnbtm"
        self.craft_IC15_model_name = "craft_ic15_20k.pth"
        self.CRAFT_GENERAL_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
        self.craft_general_model_name = "craft_mlt_25k.pth"
        self.craft_model_path = craft_model_path
        self.craft_net = self.__load_craftnet_model(self.craft_model_path)

    def __set_refine_net(self, refinenet_model_path):
        self.REFINENET_GDRIVE_URL = "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"
        self.refinenet_model_name = "craft_refiner_CTW1500.pth"
        self.refinenet_model_path = refinenet_model_path
        if not self.refinenet_model_path is None:
            self.refine_net = self.__load_refinenet_model(self.refinenet_model_path)
        else:
            self.refine_net = None

    def get_prediction(self, image=None,
                       text_threshold: float = 0.7,
                       link_threshold: float = 0.4,
                       low_text: float = 0.4,
                       square_size: int = 1280,
                       mag_ratio=1,
                       poly: bool = True,
                       only_characters=False,
                       show_time: bool = False):
        """
        Predicts bounding boxes where the text. The main function that gives bounding boxes.
        :param image: image to be processed
        :param text_threshold: text confidence threshold
        :param link_threshold: link confidence threshold
        :param low_text: text low-bound score
        :param square_size: desired longest image size for inference
        :param mag_ratio: image magnification ratio
            Default: 1
        :param poly: enable polygon type
        :param show_time: show processing time
        :return:
            {
            "masks": lists of predicted masks 2d as bool array,
            "boxes": list of coords of points of predicted boxes,
            "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
            "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
            "heatmaps": visualizations of the detected characters/links,
            "times": elapsed times of the sub modules, in seconds
            }
        :returns:
            {
            "boxes": boxes,
            "boxes_as_ratios": boxes_as_ratio,
            "polys": polys,
            "polys_as_ratios": polys_as_ratio,
            "heatmaps": {
                "text_score_heatmap": text_score_heatmap,
                "link_score_heatmap": link_score_heatmap,
            },
            "times": times,
        }
        """

        if image is None:
            image = self.image
        assert not image is None, "Image is None please enter image in numpy format or full path to load"
        image = self.set_image(image)

        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, square_size, interpolation=cv2.INTER_CUBIC, mag_ratio=mag_ratio  # old: cv2.INTER_LINEAR
        )
        ratio_h = ratio_w = 1 / target_ratio
        resize_time = time.time() - t0
        t0 = time.time()

        # preprocessing
        x = imgproc.normalize_mean_variance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
        x = x.to(self.device)
        preprocessing_time = time.time() - t0
        t0 = time.time()

        # forward pass
        with torch.no_grad():
            y, feature = self.craft_net(x)
        craftnet_time = time.time() - t0
        t0 = time.time()

        # make score and link map
        score_text = y[0, :, :, 0].detach().cpu().data.numpy()
        score_link = y[0, :, :, 1].detach().cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].detach().cpu().data.numpy()
        refinenet_time = time.time() - t0
        t0 = time.time()

        # Post-processing
        boxes, polys = craft_utils.get_detection_boxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, only_characters
        )

        # coordinate adjustment
        boxes, polys = self.coordinate_adjustment(boxes, polys, ratio_h, ratio_w)

        # get image size
        img_height = image.shape[0]
        img_width = image.shape[1]

        # calculate box coords as ratios to image size
        boxes_as_ratio = self.as_ratio(boxes, img_width, img_height)

        # calculate poly coords as ratios to image size
        polys_as_ratio = self.as_ratio(polys, img_width, img_height)

        text_score_heatmap = imgproc.cvt2HeatmapImg(score_text)
        link_score_heatmap = imgproc.cvt2HeatmapImg(score_link)

        postprocess_time = time.time() - t0

        times = {
            "resize_time": resize_time,
            "preprocessing_time": preprocessing_time,
            "craftnet_time": craftnet_time,
            "refinenet_time": refinenet_time,
            "postprocess_time": postprocess_time,
        }

        if show_time:
            print(
                "\ninfer/postproc time : {:.3f}/{:.3f}".format(
                    refinenet_time + refinenet_time, postprocess_time
                )
            )

        try:
            boxes = np.stack(boxes[:])  # multiple nested boxes arrays in to one array. TO BE ENSURE!
            boxes_as_ratio = np.stack(boxes_as_ratio[:])  # multiple nested boxes arrays in to one array. TO BE ENSURE!
        except:
            warning("boxes empty so that there is nothing to detect.")

        detection_result = {
            "boxes": boxes,  # len() -> wrong: 14, correct: 15
            "boxes_as_ratios": boxes_as_ratio,
            "polys": polys,  # all dims have to be same so that cannot stack into one array.
            "polys_as_ratios": polys_as_ratio,
            "heatmaps": {
                "text_score_heatmap": text_score_heatmap,
                "link_score_heatmap": link_score_heatmap,
            },
            "times": times,
        }

        self.detection_result = detection_result
        return detection_result

    def coordinate_adjustment(self, boxes, polys, ratio_h, ratio_w):
        boxes = craft_utils.adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjust_result_coordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]
        return boxes, polys

    def as_ratio(self, boxes=None, img_width=None, img_height=None, crop_type=None):
        """
        Gives bounding boxes as ratio.
        :param boxes: bounding boxes. if don't set default will apply.
        Note: Please set bounding box as boxes or poly.
        Also you can get from self.detection_result[crop_type] or arange_regions method
            Default; boxes = self.predicted_polygon_image
        :param img_width: image width
        :param img_height: image height
        :param crop_type: select one. boxes, boxes_as_ratios, polys, polys_as_ratios
            Default; poly
        :return: box / [img_width, img_height]
        """
        if crop_type is None:
            crop_type = self.crop_type
        if boxes is None:
            boxes = self.arange_regions(crop_type, self.detection_result)
            try:
                # I am just checking is self.detection_result in default value. I don't want it.
                assert not boxes == 0, "Please set bounding box as boxes or poly. " \
                                       "Also you can get from self.detection_result[crop_type] or arange_regions method"
            except:
                # fucking terrible idea but it works.
                pass

        if img_height is None:
            img_height = self.image.shape[0]
        if img_width is None:
            img_width = self.image.shape[1]

        # box.shape
        # Out[6]: (4, 2)
        # box / [img_width, img_height]
        # Out[7]:
        # array([[0.14765243, 0.03953583],
        #        [0.84265315, 0.05366367],
        #        [0.84215385, 0.11436041],
        #        [0.14715309, 0.10023257]])

        boxes_as_ratio = []
        for box in boxes:
            boxes_as_ratio.append(box / [img_width, img_height])
        boxes_as_ratio = np.array(boxes_as_ratio)
        return boxes_as_ratio

    def near_far_coords(self, boxes=None, crop_type=None, is_ratio=False, last_index=-2):
        """
        # O.....................
        # .(x1, y1)----(x2, y2).
        # .   |------------|   .
        # .   |------------|   .
        # .(x3, y3)----(x4, y4).
        # .....................f
        Selects (near and far coordinates from O pixel) (x1, y1) and (x4, y4) coordinates.
        Basicly it converts craft_text_detector "bounding box" system to
        handwritten_text_recognition "bounding box" system
        :param boxes: bounding boxes. if don't set default will apply.
        Note: Please set bounding box as boxes or poly.
        Also you can get from self.detection_result[crop_type] or arange_regions method
            Default; boxes = self.predicted_polygon_image
        :param crop_type: select one. boxes, boxes_as_ratios, polys, polys_as_ratios
            Default; poly
        :param is_ratio: box / [img_width, img_height]
        :param last_index: (x4, y4) index
        :return: near and far coordinates from O pixel
        """
        if crop_type is None:
            crop_type = self.crop_type
        if boxes is None:
            boxes = self.arange_regions(crop_type, self.detection_result)
            try:
                # I am just checking is self.detection_result in default value. I don't want it.
                assert not boxes == 0, "Please set bounding box as boxes or poly. " \
                                       "Also you can get from self.detection_result[crop_type] or arange_regions method"
            except:
                # fucking terrible idea but it works.
                pass
        if is_ratio:
            boxes = self.as_ratio(boxes=boxes, crop_type=crop_type)
        near_far_coords = []
        # boxes = np.stack(boxes)
        # if not isinstance(boxes, list):  # crop_type == "box":
        for idx, box in enumerate(boxes):
            # box = np.stack(box)
            try:
                # TODO! use min and max later for polygons.
                x1y1 = box[0]  # (x1, y1)
                x4y4 = box[last_index]  # (x4, y4)
            except:
                x1y1 = box[0, 0]  # (x1, y1)
                x4y4 = box[0, last_index]  # (x4, y4)
            near_far_coords.append([x1y1[0], x1y1[1], x4y4[0], x4y4[1]])  # all values in one line

        near_far_coords = np.array(near_far_coords)
        return near_far_coords

    # detect texts
    def detect_text(self, image=None,
                    output_dir='outputs/',
                    rectify=True,
                    export_extra=True,
                    text_threshold=0.7,
                    link_threshold=0.4,
                    low_text=0.4,
                    only_characters=False,
                    square_size=1280,
                    mag_ratio=1,
                    show_time=False,
                    crop_type=None):
        """
        Detects text but has some extra functionalities.
        :param image: path to the image to be processed
        :param output_dir: path to the results to be exported
        :param rectify: rectify detected polygon by affine transform
        :param export_extra: export heatmap, detection points, box visualization
        :param text_threshold: text confidence threshold
        :param link_threshold: link confidence threshold
        :param low_text: text low-bound score
        :param square_size: desired longest image size for inference
        :param mag_ratio: image magnification ratio
            Default: 1
        :param show_time: show processing time
        :param crop_type: crop regions by detected boxes or polys ("poly" or "box")
        :return:
            {
            "masks": lists of predicted masks 2d as bool array,
            "boxes": list of coords of points of predicted boxes,
            "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
            "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
            "heatmaps": visualization of the detected characters/links,
            "text_crop_paths": list of paths of the exported text boxes/polys,
            "times": elapsed times of the sub modules, in seconds
            }
        :returns:
            prediction_result
        """

        if crop_type is None:
            crop_type = self.crop_type

        # load image
        image_path = "output.jpg"
        if isinstance(image, str):
            image_path = image
        image = self.set_image(image)

        # perform prediction
        prediction_result = self.get_prediction(image=image, text_threshold=text_threshold,
                                                link_threshold=link_threshold, low_text=low_text,
                                                square_size=square_size,
                                                mag_ratio=mag_ratio, only_characters=only_characters,
                                                show_time=show_time)

        # arange regions
        regions = self.arange_regions(crop_type, prediction_result)

        # export if output_dir is given
        self.export_and_save_all(export_extra=export_extra, image_path=image_path, image=self.image,
                                 output_dir=output_dir,
                                 prediction_result=prediction_result, rectify=rectify, regions=regions)

        # return prediction results
        return prediction_result

    def get_detected_polygons(self, rectify: bool = True, crop_type: str = "poly", gray_scale=False):
        """
        Get detection region image array as numpy array
        :param rectify: do you want to rectify?
        Note: Use as_ratio method after the get_detected_polygons method.
        Gives wrong answer because ration numbers are so small
            Default; True
        :param crop_type: select one. boxes, boxes_as_ratios, polys, polys_as_ratios
            Default; poly
        :param gray_scale: If you want in grayscale set it True
            Default; False
        :return:
        """

        # arange regions
        regions = self.arange_regions(crop_type, self.detection_result)

        # export detected text regions
        # TODO! !!! Problem !!!
        # nested array
        # 'polys': array([array([[5.504257, 146.648],
        #                        [485.75977, 143.48842],
        #                        [486.35083, 233.32661],
        #                        [6.095291, 236.48619]], dtype=float32),
        #                 array([[144., 238.],
        #                        [352., 238.],
        #                        [352., 260.],
        #                        [144., 260.]], dtype=float32),
        #                 array([[430., 240.],
        #                        [510., 240.],
        #                        [510., 260.],
        #                        [430., 260.]], dtype=float32)], dtype=object),
        # not nested array.
        # 'polys_as_ratios': array([[[0.0107505, 0.42140228],
        #                            [0.94874954, 0.41232304],
        #                            [0.94990396, 0.67047878],
        #                            [0.01190487, 0.67955802]],
        #
        #                           [[0.28125, 0.68390805],
        #                            [0.6875, 0.68390805],
        #                            [0.6875, 0.74712644],
        #                            [0.28125, 0.74712644]],
        #
        #                           [[0.83984375, 0.68965517],
        #                            [0.99609375, 0.68965517],
        #                            [0.99609375, 0.74712644],
        #                            [0.83984375, 0.74712644]]]),
        if crop_type == 'boxes_as_ratios' or crop_type == 'polys_as_ratios':
            # rectify = False
            assert False, "use as_ratio method after the get_detected_polygons method. " \
                          "Gives wrong answer because ration numbers are so small"
        self.predicted_polygon_image = export_detected_polygons(
            image=self.image,  # image should come from same class
            regions=regions,
            rectify=rectify,
            gray_scale=gray_scale
        )
        return self.predicted_polygon_image

    def export_and_save_all(self, export_extra=True, image_path="output.jpg", image=None, output_dir='outputs/',
                            prediction_result=None, rectify=True, regions=None):
        prediction_result["text_crop_paths"] = []
        if output_dir is not None:
            # export detected text regions
            exported_file_paths = export_detected_regions(
                image_path=image_path,
                image=image,
                regions=regions,
                output_dir=output_dir,
                rectify=rectify,
            )
            prediction_result["text_crop_paths"] = exported_file_paths

            # export heatmap, detection points, box visualization
            if export_extra:
                export_extra_results(
                    image_path=image_path,
                    image=image,
                    regions=regions,
                    heatmaps=prediction_result["heatmaps"],
                    output_dir=output_dir,
                )

    def arange_regions(self, crop_type=None, prediction_result=None):
        # "boxes": boxes,  # len() -> wrong: 14, correct: 15
        # "boxes_as_ratios": boxes_as_ratio,
        # "polys": polys,
        # "polys_as_ratios": polys_as_ratio,
        if crop_type is None:
            crop_type = self.crop_type
        if prediction_result is None:
            prediction_result = self.detection_result

        if crop_type == "box":
            regions = prediction_result["boxes"]  # boxes
        elif crop_type == "poly":
            regions = prediction_result["polys"]  # polys
        else:
            raise TypeError("crop_type can be only 'polys' or 'boxes'")
        return regions

    def __load_state_dict(self, net, weight_path):
        """
        1) Loads weights and biases.
        2) Deserialize them.
        3) Transport to device
        4) Make it pytorch "dataparallel"
        5) Turn it into evaluation mode.
        6) Return it.
        :param net: Artificial Neural network(model) that makes main job
        :param weight_path: Serialized pth file path with name
        :return: loaded network
        """
        net.load_state_dict(copyStateDict(torch.load(weight_path)))

        net = net.to(self.device)
        net = torch.nn.DataParallel(net)
        net.eval()
        return net

    def __load_craftnet_model(self, craft_model_path=None):
        """
        Loads craftnet network(model)
        :param craft_model_path: Serialized craftnet network(model) file path with name
        :return: loaded network
        """
        # get craft net path
        weight_path = get_weight_path(craft_model_path,
                                      self.CRAFT_GENERAL_GDRIVE_URL,
                                      self.craft_general_model_name)
        # arange device
        craft_net = self.__load_state_dict(self.craft_net, weight_path)
        return craft_net

    def __load_refinenet_model(self, refinenet_model_path=None):
        """
        Loads refinenet network(model)
        :param refinenet_model_path: Serialized refinenet network(model) file path with name
        Refiner network eliminates low probability detections.
            Default: None
        :return: loaded network
        """
        # get refine net path
        weight_path = get_weight_path(refinenet_model_path,
                                      self.REFINENET_GDRIVE_URL,
                                      "craft_refiner_CTW1500.pth")

        # arange device
        refine_net = self.__load_state_dict(self.refine_net, weight_path)
        return refine_net


if __name__ == "__main__":
    # set image path and export folder directory
    # image_name = "a8.png"
    # image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\IAM8" + "/" + image_name
    # image_name = "htr_level_5.jpg"
    # image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\test_images2" + "/" + image_name
    # image_name = 'idcard.png'
    # image_path = '../figures/' + image_name
    # image_name = 'plate1.jpg'
    # image_path = r'C:/Users/selcu/PycharmProjects/ocr_toolkit/license_plate_images/' + image_name
    image_name = "positive2.png"
    image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\license_plate_images\plates" + "/" + image_name

    output_dir = image_name + '/'


    def image_preprocess(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # plt.imshow(gray, cmap="gray")
        # plt.show()

        ## not working
        # makes worse
        # blur = cv2.medianBlur(gray, 1)
        # plt.imshow(blur, cmap="gray")
        # plt.show()

        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11,
                                         C=1)  # C=2
        # plt.imshow(adaptive, cmap="gray")
        # plt.show()
        adaptive_res = cv2.bitwise_and(image, image, mask=adaptive)

        # second thresholding option
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([0, 0, 0])
        upper_blue = np.array([180, 180, 180])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        residual = cv2.bitwise_and(image, image, mask=mask)
        # plt.imshow(mask, cmap="gray")
        # plt.show()

        ## Not working
        # morphological transformation
        # kernel = np.ones((5, 5), np.uint8)
        # erosion = cv2.erode(adaptive, kernel, iterations=2)
        # plt.imshow(erosion, cmap="gray")
        # plt.show()
        #
        # opening = cv2.morphologyEx(adaptive, cv2.MORPH_ELLIPSE, kernel)
        # plt.imshow(opening, cmap="gray")
        # plt.show()

        ## not working
        # kernel_sharpening = np.array([[-1, -1, -1],
        #                               [-1, 9, -1],
        #                               [-1, -1, -1]])
        # sharpened = cv2.filter2D(mask, -1, kernel_sharpening)
        # plt.imshow(sharpened, cmap="gray")
        # plt.show()

        # adaptive_res_expd = np.dstack([adaptive_res] * 3)
        # res_expd = np.dstack([res] * 3)
        return adaptive_res, residual


    def test_oops(image_path, output_dir):
        craft_model_path = "../craft_mlt_25k.pth"
        refinenet_model_path = "../craft_refiner_CTW1500.pth"
        show_time = False
        # read image
        image = read_image(image_path)
        # not working well with preprocess
        # adaptive_res, residual = image_preprocess(image)
        # image = residual
        # create craft_detector class
        pred = craft_detector(image=image,
                              craft_model_path=craft_model_path,
                              refinenet_model_path=refinenet_model_path,
                              device="gpu")
        only_characters = True
        prediction_result = pred.detect_text(image=image_path, output_dir=output_dir, rectify=True, export_extra=False,
                                             text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                                             only_characters=only_characters, square_size=720, show_time=show_time,
                                             crop_type="poly")
        print(len(prediction_result[
                      "boxes"]))  # refinenet_model_path=None -> 51, refinenet_model_path=refinenet_model_path -> 19
        print(len(prediction_result["boxes"][0]))  # 4
        print(len(prediction_result["boxes"][0][0]))  # 2
        print(int(prediction_result["boxes"][0][0][0]))  # 115
        # perform prediction
        # TODO! tune parameters
        prediction_result = pred(image=image,
                                 text_threshold=0.7,  # 0.7
                                 link_threshold=0.4,  # 0.4
                                 low_text=0.4,  # 0.6, 0.4
                                 square_size=1280,
                                 show_time=True,
                                 only_characters=only_characters)
        # export detected text regions
        exported_file_paths = export_detected_regions(
            image_path=image_path,
            image=image,
            regions=prediction_result["boxes"],
            output_dir=output_dir,
            rectify=True
        )
        # export heatmap, detection points, box visualization
        export_extra_results(
            image_path=image_path,
            image=image,
            regions=prediction_result["boxes"],
            heatmaps=prediction_result["heatmaps"],
            output_dir=output_dir
        )

        crop_type = "box"  # poly
        print(pred.near_far_coords(crop_type=crop_type))
        # pprint(prediction_result)


    # Best time without refiner: 0.252/0.171
    # Best time with refiner: 0.138/0.039. Wow!!!
    test_oops(image_path, output_dir)
