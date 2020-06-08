import sys
from os import path


dir_path = path.dirname(path.abspath(__file__)) + "/"
sys.path.append(dir_path)
try:
    from darknet import Darknet
except:
    try:
        from Darknet.darknet import Darknet
    except:
        try:
            from LPR.Darknet.darknet import Darknet
        except:
            from detection.LPR.Darknet.darknet import Darknet

try:
    import dknet_label_conversion
    import nms
    import ModelLoadError, ReadImageError
except:
    try:
        from platerecognition.src.label import dknet_label_conversion
        from platerecognition.src.utils import nms
        from platerecognition.exceptions import ModelLoadError, ReadImageError
    except:
        from detection.LPR.platerecognition.src.label import dknet_label_conversion
        from detection.LPR.platerecognition.src.utils import nms
        from detection.LPR.platerecognition.exceptions import ModelLoadError, ReadImageError


class OCR:
    """
    OCR class is responsible to find licence plate text from plate images

    Attributes:
        dark(Darknet object): Darknet object to use yolov3 model

    Methods:
        detect(img_path): Detection licence plate text using img_path
    """

    def __init__(self, use_gpu=False):
        __ocr_weights = (dir_path + 'lpr-models/ocr-net.weights')
        __ocr_netcfg = (dir_path + 'lpr-models/ocr-net.cfg')
        __ocr_dataset = (dir_path + 'lpr-models/ocr-net.data')
        __ocr_names = (dir_path + 'lpr-models/ocr-net.names')

        self.__ocr_threshold = .4
        self.temp_path = dir_path + "plate.png"  ## This image is temporary. Darknet need path to use Detect function. Processed image will be saved on this path and then darknet will use this path again.

        ##Load OCR model
        try:
            self.dark = Darknet(__ocr_netcfg,
                                __ocr_weights,
                                __ocr_dataset,
                                __ocr_names,
                                gpu=use_gpu)
        except Exception as e:
            raise ModelLoadError
        print("\n[INFO] OCR model succesfully loaded...\n\n")

    def detect(self, frame):
        ''' Detection to find licence plate text

        Parameters:
            frame(str): Path of image or RGB image

        Return:
            lp_str(str): Licence plate text
        '''

        if isinstance(frame, str):
            frame = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_RGB2BGR)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        ##Detect characters from plate(s) image(s)
        res = self.dark.detect(frame,
                               thresh=self.__ocr_threshold,
                               nms=None)

        if isinstance(frame, str):
            try:
                w, h = cv2.imread(frame).shape[:2]
            except Exception as e:
                raise ReadImageError
        else:
            w, h = frame.shape[:2]

        ##if characters found on label
        if len(res):
            L = dknet_label_conversion(res, w, h)
            L = nms(L, .45)

            ##sort character orders
            L.sort(key=lambda x: x.tl()[0])

            # if square_plate:
            # lower = [l for l in L if l.tl()[1]<-0.009]
            # higher = [l for l in L if l.tl()[1]>=-0.009]
            # lower.sort(key=lambda x: x.tl()[0])
            # higher.sort(key=lambda x: x.tl()[0])
            # L = lower+higher

            lp_str = ''.join([chr(l.cl()) for l in L])
            return lp_str
        else:  ##if there is no plate characters on plate(OCR couldn't find any char)
            return ""


if __name__ == '__main__':
    import sys

    ocr = OCR(use_gpu=True)
    import time
    import cv2

    image_name = "data/track0061[01].png"  # sys.argv[1]
    frame = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_RGB2BGR)
    count = int(sys.argv[2])
    start_time = time.time()
    for _ in range(count):
        result = ocr.detect(frame)
    print(f"OCR took: {(time.time() - start_time) / count}")
    print(result)
