# Standart modules

# Well known 3thd party modules
# Detection modules
# Detection - craft modules
from pprint import pprint

from detection.craft_text_detector.craft_text_detector import craft_detector

# Recognition modules
# Recognition - handwritten_text_recognition modules
from detection.craft_text_detector.craft_text_detector.file_utils import export_detected_polygons
from detection.craft_text_detector.craft_text_detector.imgproc import read_image
from detection.detection_selector import detection_selector
from recognition.handwritten_text_recognition.recognition.utils import device_selection_helper
from recognition.handwritten_text_recognition.recognition.recognizer import recognize

# Recognition - minimal_text_recognition modules
from recognition.recognition_selector import recognition_selector


class selector_pipeline:
    def __init__(self, image, rectify: bool = True, text_threshold: float = 0.7, link_threshold: float = 0.4,
                 low_text: float = 0.4,
                 square_size: int = 1280, show_time: bool = False, crop_type: str = "poly", detection_model_paths=None,
                 recognition_model_paths=None, detection_switch: str = "craft", recognition_switch: str = "handwritten",
                 mag_ratio: int = 1, num_device: int = 1, device: str = "gpu", language="eng"):
        self.detection_result = []  # PYTORCH ;)
        self.recognition_result = []  # MXNET ;)

        # detection networks
        # "craft_mlt_25k.pth"
        # "craft_refiner_CTW1500.pth"
        self.detection_selector = detection_selector(image=image, detection_switch=detection_switch, device=device)

        # recognition networks
        # net_parameter_pathname = [
        #     net_parameter_path+"denoiser2.params",
        #     net_parameter_path+"handwriting_line8.params",
        #     net_parameter_path+"paragraph_segmentation2.params",
        #     net_parameter_path+"word_segmentation2.params",
        # ]
        self.recognition_selector = recognition_selector(image=image, num_device=num_device,
                                                         recognition_model_paths=recognition_model_paths,
                                                         recognition_switch=recognition_switch,
                                                         device="cpu",
                                                         language=language)  # TODO! recognition is too big to fit in to gpu.

        # hyperparameters
        self.reload_hyperparameters(rectify=rectify, text_threshold=text_threshold, link_threshold=link_threshold,
                                    low_text=low_text, square_size=square_size, show_time=show_time,
                                    detection_model_paths=detection_model_paths,
                                    recognition_model_paths=recognition_model_paths,
                                    crop_type=crop_type,
                                    detection_switch=detection_switch, recognition_switch=recognition_switch,
                                    mag_ratio=mag_ratio, num_device=num_device, device=device, language=language)

        self.image = self.detection_selector.image
        self.detection_result = 0
        self.predicted_bb = 0
        self.recognition_result = 0

    def reload_hyperparameters(self, rectify: bool = True, text_threshold: float = 0.7,
                               link_threshold: float = 0.4, low_text: float = 0.4,
                               square_size: int = 1280, show_time: bool = False, crop_type: str = "poly",
                               detection_model_paths=None,
                               recognition_model_paths=None, detection_switch: str = "craft",
                               recognition_switch: str = "handwritten",
                               mag_ratio: int = 1, num_device: int = 1, device: str = "gpu", language="eng"):
        self.detection_selector.reload_hyperparameters(rectify=rectify, text_threshold=text_threshold,
                                                       link_threshold=link_threshold,
                                                       low_text=low_text, square_size=square_size, show_time=show_time,
                                                       detection_model_paths=detection_model_paths, crop_type=crop_type,
                                                       detection_switch=detection_switch,
                                                       mag_ratio=mag_ratio, device=device)

        self.recognition_selector.reload_hyperparameters(device, num_device, recognition_switch=recognition_switch,
                                                         recognition_model_paths=recognition_model_paths,
                                                         language=language)

    def creatation_switch(self, image, detection_switch=None, recognition_switch=None,
                          detection_model_paths=None, recognition_model_paths=None, detection_device=None,
                          recognition_device=None):
        self.detection_selector.creatation_switch(image=image,
                                                  detection_switch=detection_switch,
                                                  detection_model_paths=detection_model_paths,
                                                  device=detection_device)

        self.recognition_selector.creatation_switch(image=image,
                                                    recognition_switch=recognition_switch,
                                                    recognition_model_paths=recognition_model_paths,
                                                    device=recognition_device)

    def __call__(self, *args, **kwargs):
        return self.make_it(**kwargs)

    def make_it(self, **kwargs):
        # return self.make_switch(**kwargs)
        return self.handwritten_detect_and_recognize(**kwargs)

    def make_switch(self, detection_switch=None, recognition_switch=None, **kwargs):
        if detection_switch is None:
            detection_switch = self.detection_selector.detection_switch
        if recognition_switch is None:
            recognition_switch = self.recognition_selector.recognition_switch

        detection_result = self.detection_selector.make_switch(detection_switch=detection_switch, **kwargs)
        line_images_array = self.only_detected_polygons(rectify=self.detection_selector.rectify,
                                                        crop_type=self.detection_selector.crop_type,
                                                        gray_scale=self.recognition_selector.net.gray_scale)

        recognition_result = self.recognition_selector.make_switch(recognition_switch=recognition_switch,
                                                                   line_images_array=line_images_array,
                                                                   language=self.recognition_selector.language,
                                                                   )

        return detection_result, recognition_result

    def handwritten_detect_and_recognize(self, image=None, crop_type="box", rectify=True, is_ratio=True, last_index=-2):
        """
        Detection algorithm predicts bounding boxes and gives to recognizer algorithm
        recognizer uses predicted bounding boxes and recognizes the handwritten.
        """
        if image is None:
            image = self.image
        self.predicted_bb = self.only_detect_bb(image=image, crop_type=crop_type,
                                                is_ratio=is_ratio, last_index=last_index)
        line_images_array = self.only_detected_polygons(rectify=rectify, crop_type=crop_type,
                                                        gray_scale=self.recognition_selector.net.gray_scale)
        self.recognition_result = self.only_recognize(line_images_array=line_images_array)
        decoded_line_ams, decoded_line_bss, decoded_line_denoisers = self.recognition_result
        decoded = {
            "decoded_line_ams": decoded_line_ams,
            "decoded_line_bss": decoded_line_bss,
            "decoded_line_denoisers": decoded_line_denoisers,
        }
        result = {
            "predicted_bb": self.predicted_bb,
            "decoded": decoded
        }
        return result

    def only_detect_bb(self, image=None, crop_type="box", is_ratio=True, last_index=-2):
        """
        Predicts bounding boxes.
        """
        # "boxes": boxes,  # len() -> wrong: 14, correct: 15
        # "boxes_as_ratios": boxes_as_ratio,
        # "polys": polys,
        # "polys_as_ratios": polys_as_ratio,
        if image is None:
            image = self.image
        self.detection_result = self.detection_selector.make_it(image=image)
        regions = self.detection_selector.get_detected_bb(crop_type=crop_type)
        regions = self.detection_selector.near_far_coords(boxes=regions, crop_type=crop_type, is_ratio=is_ratio,
                                                          last_index=last_index)  # makes coords as ratio
        self.predicted_bb = regions
        return regions

    def only_detected_polygons(self, rectify=True, crop_type=None, gray_scale=False):
        predicted_polygon_image = self.detection_selector.get_detected_polygons(rectify=rectify, crop_type=crop_type,
                                                                                gray_scale=gray_scale)
        return predicted_polygon_image

    def only_recognize(self, line_images_array=None, language="eng"):
        """
        Recognizes handwritten inside of bounding boxes.
        """
        # recognition_result = self.recognition_selector.handwritten_text_only_recognition(line_images_array=line_images_array)
        self.recognition_result = self.recognition_selector(line_images_array=line_images_array, language=language)
        return self.recognition_result

    def detect_text(self, image=None,
                    output_dir=None,
                    rectify=True,
                    export_extra=True,
                    text_threshold=0.7,
                    link_threshold=0.4,
                    low_text=0.4,
                    square_size=1280,
                    mag_ratio=1,
                    show_time=False,
                    crop_type=None):
        return self.detection_selector.detect_text(image=image,
                                                   output_dir=output_dir,
                                                   rectify=rectify,
                                                   export_extra=export_extra,
                                                   text_threshold=text_threshold,
                                                   link_threshold=link_threshold,
                                                   low_text=low_text,
                                                   square_size=square_size,
                                                   mag_ratio=mag_ratio,
                                                   show_time=show_time,
                                                   crop_type=crop_type)


if __name__ == "__main__":
    # image_name = "a1.png"
    # image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\IAM8" + "/" + image_name
    image_name = "handwritten_english.jpeg"
    image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\EnglishHandwritten" + "/" + image_name
    # image_name = "htr_level_5.jpg"
    # image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\test_images2" + "/" + image_name
    # image_name = "elyaz2.jpeg"
    # image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\TurkishHandwritten" + "/" + image_name
    # image_name = "plate1.jpg"
    # image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\license_plate_images" + "/" + image_name
    detection_switch = "craft"
    detection_model_paths = []
    detection_model_paths.append("craft_mlt_25k.pth")
    detection_model_paths.append("craft_refiner_CTW1500.pth")

    recognition_model_paths = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\recognition\handwritten_text_recognition\recognition\models/"  # windows
    recognition_switch = "handwritten"  # "handwritten"

    crop_type = "box"  # poly
    output_dir = image_name + '/'
    show_time = False
    language = "tur"
    pipeline = selector_pipeline(image_path, detection_switch=detection_switch, recognition_switch=recognition_switch,
                                 detection_model_paths=detection_model_paths,
                                 recognition_model_paths=recognition_model_paths,
                                 crop_type=crop_type,
                                 device="gpu",
                                 language=language)

    # detection_result = pipeline.make_it()
    # # pprint(detection_result)
    # detection_and_recognition = pipeline.handwritten_detect_and_recognize()
    detection_and_recognition = pipeline.make_switch()
    pprint(detection_and_recognition)

    # prediction_result = pipeline.detect_text(image=image_path, output_dir=output_dir, rectify=True,
    #                                          export_extra=True,
    #                                          text_threshold=0.7, link_threshold=0.4, low_text=0.4, square_size=720,
    #                                          show_time=show_time, crop_type=crop_type)
