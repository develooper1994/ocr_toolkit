from pprint import pprint

import numpy as np
import mxnet as mx

from recognition.handwritten_text_recognition.recognition.recognizer import recognize as handwritten_recognize
from recognition.handwritten_text_recognition.recognition.utils import device_selection_helper
from detection.craft_text_detector.craft_text_detector import craft_detector

from recognition.tesseract_text_recognition.recognition.recognizer import recognize as tes_recognize


class recognition_selector:
    def __init__(self, image, line_images_array=None, num_device=1, show=False, recognition_model_paths=None,
                 recognition_switch: str = "handwritten", device: str = "gpu", language="eng"):
        # MXNET ;)
        self.recognition_result = []

        self.reload_hyperparameters(device, num_device, line_images_array=line_images_array,
                                    recognition_switch=recognition_switch, show=show,
                                    recognition_model_paths=recognition_model_paths, language=language)

        self.net = self.creatation_switch(image=image, recognition_switch=self.recognition_switch,
                                          recognition_model_paths=self.recognition_model_paths,
                                          device=self.device)
        self.image = self.net.image

    def reload_hyperparameters(self, device, num_device, recognition_switch, line_images_array=None,
                               show=False, recognition_model_paths=None, language="eng"):
        # image = mx.image.imread("tests/TurkishHandwritten/elyaz2.jpeg")
        self.num_device = num_device
        self.device = device_selection_helper(device=device.lower(), num_device=num_device)
        self.line_images_array = line_images_array
        self.show = show
        self.language = language

        # Model paths
        self.recognition_switch = recognition_switch
        if recognition_model_paths is None:
            # paths = "recognition/handwritten_text_recognition/recognition/models/"  # linux
            paths = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\recognition\handwritten_text_recognition\recognition\models/"  # windows
            recognition_model_paths = paths
        self.recognition_model_paths = recognition_model_paths

    def creatation_switch(self, image, recognition_switch, recognition_model_paths, device):
        # TODO! not a good solution for interface changing
        recognition_switch = recognition_switch.lower()
        if recognition_switch == "handwritten":
            # MXNET ;)
            net = handwritten_recognize(image, net_parameter_path=recognition_model_paths, device=device, show=self.show)
        elif recognition_switch == "minimal":
            # PYTORCH ;)
            raise NotImplementedError
        elif recognition_switch == "tesseract":
            # tesseract, no framework ;)
            net = tes_recognize(image, language=self.language)

        else:
            net = None  # TODO! throws error
            print("There is no such algorithm. Please provide it(or help me).")

        return net

    def __call__(self, *args, **kwargs):
        return self.make_it(**kwargs)

    def make_it(self, **kwargs):
        return self.make_switch(**kwargs)

    def make_switch(self, recognition_switch=None, line_images_array=None, language="eng"):  # , **kwargs
        recognition_result = None
        if recognition_switch is not None:
            recognition_switch = recognition_switch.lower()
        else:
            recognition_switch = self.recognition_switch.lower()
        if recognition_switch == "handwritten":
            # # MXNET ;)
            # recognition_result = self.handwritten_text_recognition(**kwargs)
            recognition_result = self.handwritten_text_only_recognition(line_images_array=line_images_array)
        elif recognition_switch == "minimal":
            # PYTORCH ;)
            raise NotImplementedError
        elif recognition_switch == "tesseract":
            recognition_result = self.tesseract_text_only_recognition(line_images_array=line_images_array, language=language)
        else:
            net = None  # TODO! throws error after some operation
            print("There is no such algorithm. Please provide it(or help me).")

        return recognition_result

    def handwritten_text_only_recognition(self, line_images_array=None):
        # self.net.predicted_bb = predicted_bb
        if line_images_array is None:
            line_images_array = self.image
        # self.line_images_array = [[array([[241, 243, 242, ..., 247, 250, 250],
        #          [248, 246, 248, ..., 247, 248, 247],
        #          [246, 247, 247, ..., 249, 243, 247],
        #          ...,
        #          [246, 247, 243, ..., 229, 232, 233],
        #          [245, 243, 241, ..., 234, 236, 237],
        #          [244, 244, 242, ..., 237, 241, 239]], dtype=uint8)]]

        # perform prediction
        # line_images_array = self.net.word_to_line()
        line_images_array = [line_images_array]
        self.net.handwriting_recognition_probs(line_images_array=line_images_array)
        decoded = self.net.make_decoded()

        # self.recognition_result = line_images_array, character_probs, decoded
        self.recognition_result = decoded

        return self.recognition_result

    def minimal_text_recognition(self):
        raise NotImplementedError

    def tesseract_text_only_recognition(self, line_images_array=None, language="eng"):
        if line_images_array is None:
            line_images_array = self.image
        line_images_array = [line_images_array]
        decoded = self.net.one_step(line_images_array=line_images_array, language=language)

        self.recognition_result = decoded["decoded"]

        return self.recognition_result


if __name__ == "__main__":
    image_name = "a1.png"
    image_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\detection\craft_text_detector\figures\IAM8" + "/" + image_name
    image = mx.image.imread(image_path)
    image = image.asnumpy()

    # recognition_model_paths = "recognition/handwritten_text_recognition/recognition/models/"  # linux
    recognition_model_paths = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\recognition\handwritten_text_recognition\recognition\models/"  # windows

    language = "tur"
    # recog_selector = recognition_selector(image=image, num_device=1, recognition_model_paths=recognition_model_paths,
    #                                       recognition_switch="handwritten", device="cpu", language=language)

    recog_selector = recognition_selector(image="htr_level_5.jpg", num_device=1, recognition_model_paths=recognition_model_paths,
                                          recognition_switch="tesseract", device="cpu", language=language)

    recognition_result = recog_selector.make_it()
    pprint(recognition_result)
