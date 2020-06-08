from pprint import pprint

from detection.craft_text_detector.craft_text_detector import craft_detector
from detection.craft_text_detector.craft_text_detector.file_utils import export_detected_polygons

# helps to select detection algorithm
from recognition.handwritten_text_recognition.recognition.utils import device_selection_helper


class detection_selector:
    def __init__(self, image, rectify=True, text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                 square_size=1280, show_time=False, crop_type="poly", detection_model_paths=None,
                 detection_switch: str = "craft",
                 mag_ratio=1, device: str = "gpu"):
        # PYTORCH ;)
        self.detection_result = []

        # hyperparameters
        self.reload_hyperparameters(rectify=rectify, text_threshold=text_threshold, link_threshold=link_threshold,
                                    low_text=low_text, square_size=square_size, show_time=show_time,
                                    detection_model_paths=detection_model_paths, crop_type=crop_type,
                                    detection_switch=detection_switch,
                                    mag_ratio=mag_ratio, device=device)

        self.net = self.creatation_switch(image=image, detection_switch=self.detection_switch,
                                          detection_model_paths=self.detection_model_paths,
                                          device=self.device)
        self.image = self.net.image
        self.predicted_bb = 0

    def reload_hyperparameters(self, rectify=True, text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                               square_size=1280, show_time=False, crop_type="poly", detection_model_paths=None,
                               detection_switch: str = "craft", mag_ratio=1, device: str = "gpu"):
        self.rectify = rectify
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.square_size = square_size
        self.show_time = show_time
        self.crop_type = crop_type

        self.detection_switch = detection_switch.lower()
        self.mag_ratio = mag_ratio
        self.device = device_selection_helper(device=device.lower(), framework="pytorch")

        # Model paths
        self.detection_model_paths = detection_model_paths
        if detection_model_paths is None:
            detection_model_paths = []
            detection_model_paths.append("../craft_mlt_25k.pth")
            detection_model_paths.append("../craft_refiner_CTW1500.pth")
            self.detection_model_paths = detection_model_paths

    def creatation_switch(self, image, detection_switch, detection_model_paths, device):
        detection_switch = detection_switch.lower()
        if detection_switch == "craft":
            craft_model_path = detection_model_paths[0]  # "../craft_mlt_25k.pth"
            refinenet_model_path = detection_model_paths[1]  # "../craft_refiner_CTW1500.pth"
            # PYTORCH ;)
            net = craft_detector.craft_detector(image=image,
                                                craft_model_path=craft_model_path,
                                                refinenet_model_path=refinenet_model_path,
                                                device=device)
        # elif detection_switch == "another_detection_algorithm_object":
        #     pass
        else:
            net = None  # TODO! throws error
            print("There is no such algorithm. Please provide it(or help me).")

        return net

    def __call__(self, *args, **kwargs):
        return self.make_it(**kwargs)

    def make_it(self, **kwargs):
        return self.make_switch(**kwargs)

    def make_switch(self, detection_switch=None, **kwargs):
        detection_result = None
        if detection_switch is not None:
            detection_switch = detection_switch.lower()
        else:
            detection_switch = self.detection_switch.lower()
        if detection_switch == "craft":
            # PYTORCH ;)
            detection_result = self.net(**kwargs)
        # elif detection_switch == "another_detection_algorithm":
        #     pass
        else:
            net = None  # TODO! throws error
            print("There is no such algorithm. Please provide it(or help me).")

        self.detection_result = detection_result
        return detection_result

    # def another_detection_algorithm(self):
    #     pass
    def as_ratio(self, boxes=None, img_width=None, img_height=None, crop_type=None):
        return self.net.as_ratio(boxes=boxes, img_width=img_width, img_height=img_height, crop_type=crop_type)

    def get_detected_polygons(self, rectify=True, crop_type=None, gray_scale=False):
        return self.net.get_detected_polygons(rectify=rectify, crop_type=crop_type, gray_scale=gray_scale)

    def get_detected_bb(self, crop_type=None):
        if crop_type is None:
            crop_type = self.crop_type
        regions = self.net.arange_regions(crop_type, self.detection_result)
        # regions = self.net.as_ratio(boxes=regions, crop_type=crop_type)
        # regions = self.near_far_coords(boxes=regions, crop_type=crop_type, is_ratio=is_ratio,
        #                                last_index=last_index)  # makes coords as ratio
        return regions

    def near_far_coords(self, boxes=None, crop_type=None, is_ratio=False, last_index=-2):
        return self.net.near_far_coords(boxes=boxes, crop_type=crop_type, is_ratio=is_ratio, last_index=last_index)

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
        return self.net.detect_text(image=image,
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
    # image_name = "htr_level_5.jpg"
    # image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\test_images2" + "/" + image_name
    # image_name = "plate1.jpg"
    # image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\license_plate_images/" + image_name
    image_name = "htr_level_5.jpg"
    image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\test_images2" + "/" + image_name
    selector = "craft"
    detection_model_paths = []
    detection_model_paths.append("craft_mlt_25k.pth")
    detection_model_paths.append("craft_refiner_CTW1500.pth")

    # recognition_model_paths = "recognition/handwritten_text_recognition/recognition/models/"  # linux
    recognition_model_paths = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\recognition\handwritten_text_recognition\recognition\models/"  # windows

    dev_selector = detection_selector(image_path, detection_switch=selector,
                                      detection_model_paths=detection_model_paths, device="gpu")

    # "boxes": boxes,  # len() -> wrong: 14, correct: 15
    # "boxes_as_ratios": boxes_as_ratio,
    # "polys": polys,
    # "polys_as_ratios": polys_as_ratio,

    crop_type = "box"  # poly
    detection_result = dev_selector.make_it()
    # pprint(detection_result)
    predicted_bb = dev_selector.get_detected_bb(crop_type=crop_type)
    # pprint(predicted_bb)
    predicted_bb_as_ratio = dev_selector.get_detected_bb(crop_type=crop_type)
    # pprint(predicted_bb_as_ratio)

    output_dir = image_name + '/'
    show_time = False
    prediction_result = dev_selector.detect_text(image=image_path, output_dir=output_dir, rectify=True,
                                                 export_extra=True,
                                                 text_threshold=0.7, link_threshold=0.4, low_text=0.4, square_size=720,
                                                 show_time=show_time, crop_type=crop_type)

    pprint(prediction_result)
