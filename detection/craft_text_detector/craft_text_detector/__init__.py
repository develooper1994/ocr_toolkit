# TODO! rewrite documentation.
from __future__ import absolute_import

__version__ = "0.2.3"
__author__ = "Mustafa Selçuk Çağlar"

try:
    from . import craft_detector
    from . import craft_utils
    from . import expand_bounding_box
    from . import file_utils
    from . import imgproc
    from . import word_to_line
except:
    try:
        from ..craft_text_detector import craft_detector
        from ..craft_text_detector import craft_utils
        from ..craft_text_detector import expand_bounding_box
        from ..craft_text_detector import file_utils
        from ..craft_text_detector import imgproc
        from ..craft_text_detector import word_to_line
    except:
        from detection.craft_text_detector.craft_text_detector import craft_detector
        from detection.craft_text_detector.craft_text_detector import craft_utils
        from detection.craft_text_detector.craft_text_detector import expand_bounding_box
        from detection.craft_text_detector.craft_text_detector import file_utils
        from detection.craft_text_detector.craft_text_detector import imgproc
        from detection.craft_text_detector.craft_text_detector import word_to_line


def detect_text(image, output_dir=None, rectify=True, export_extra=True,
                text_threshold=0.7, link_threshold=0.4,
                low_text=0.4, long_size=1280,
                device="cpu", show_time=False,
                crop_type="poly"):
    """
    Detects text but has some extra functionalities.
    :param image: path to the image to be processed
    :param output_dir: path to the results to be exported
    :param rectify: rectify detected polygon by affine transform
    :param export_extra: export heatmap, detection points, box visualization
    :param text_threshold: text confidence threshold
    :param link_threshold: link confidence threshold
    :param low_text: text low-bound score
    :param long_size: desired longest image size for inference
    :param show_time: show processing time
    :param cuda: cuda switch
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
    # load craft model
    craft_net = craft_detector.craft_detector(image=image, device=device)

    prediction_result = craft_net.detect_text(image=image, output_dir=output_dir, rectify=rectify,
                                              export_extra=export_extra, text_threshold=text_threshold,
                                              link_threshold=link_threshold, low_text=low_text, square_size=long_size,
                                              show_time=show_time, crop_type=crop_type)

    # return prediction results
    return prediction_result

