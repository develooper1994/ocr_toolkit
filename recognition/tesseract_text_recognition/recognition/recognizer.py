import pytesseract as pytes

from detection.craft_text_detector.craft_text_detector.imgproc import read_image

pytes.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract_ocr\tesseract.exe"
from PIL import Image


class recognize:
    """
    Only successful for PDF, id cards, credit and bank cards and look like digital written.
    """
    def __init__(self, image, language="eng"):
        self.reload(image, language)

    def reload(self, image, language="eng"):
        self.line_images_array = self.set_image(image)
        self.language = language
        self.gray_scale = True

    def __call__(self, *args, **kwargs):
        return self.one_step()

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

    def one_step(self, **kwargs):
        """
        Calculate all in of them in one (long) step
        :return: all calculated results.
            results = {
                'decoded': decoded
            }
        """
        # recognition
        decoded = self.make_recognition(**kwargs)
        # decoded_line_ams, decoded_line_bss, decoded_line_denoisers = decoded
        all_results = {
            'decoded': decoded
        }
        return all_results

    def make_recognition(self, line_images_array=None, language=None):
        decoded = self.make_decoded(line_images_array, language)
        return decoded

    def make_decoded(self, line_images_array=None, language=None):
        if line_images_array is None:
            line_images_array = self.line_images_array

        if language is None:
            language = self.language

        form_decoded_strings = []
        try:
            for line_image in line_images_array:  # if you want to give number for each line_images_array use enumurate.
                decoded_string = pytes.image_to_string(image=line_image, lang=language)
                form_decoded_strings.append(decoded_string)
        except:  # TypeError: Unsupported image object
            for line_image in line_images_array:
                for line in line_image:  # more than one.
                    decoded_string = pytes.image_to_string(image=line, lang=language)
                    form_decoded_strings.append(decoded_string)
        return form_decoded_strings


if __name__ == "__main__":
    image_name = 'htr_level_5.jpg'
    image = [Image.open(image_name)]
    language = "tur"
    recog = recognize(image, language=language)

    print(recog())
