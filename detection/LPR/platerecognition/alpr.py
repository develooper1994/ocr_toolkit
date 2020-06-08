import sys
from os import path

dir_path = path.dirname(path.abspath(__file__)) + "/"
sys.path.append(dir_path)
import cv2
import numpy as np
from time import time

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
    import Label, Shape
    import crop_region, im2single
    import load_model, detect_lp
    import OCR
    import check_standart, get_coords, remove_duplicated_plates
    import OutOfBoundsError, ModelLoadError, ReadImageError, WriteImageError
except:
    try:
        from platerecognition.src.label import Label, Shape
        from platerecognition.src.utils import crop_region, im2single
        from platerecognition.src.keras_utils import load_model, detect_lp
        from platerecognition.ocr import OCR
        from platerecognition.utils import check_standart, get_coords, remove_duplicated_plates
        from platerecognition.exceptions import OutOfBoundsError, ModelLoadError, ReadImageError, WriteImageError
    except:
        try:
            from LPR.platerecognition.src.label import Label, Shape
            from LPR.platerecognition.src.utils import crop_region, im2single
            from LPR.platerecognition.src.keras_utils import load_model, detect_lp
            from LPR.platerecognition.ocr import OCR
            from LPR.platerecognition.utils import check_standart, get_coords, remove_duplicated_plates
            from LPR.platerecognition.exceptions import OutOfBoundsError, ModelLoadError, ReadImageError, \
                WriteImageError
        except:
            from detection.LPR.platerecognition.src.label import Label, Shape
            from detection.LPR.platerecognition.src.utils import crop_region, im2single
            from detection.LPR.platerecognition.src.keras_utils import load_model, detect_lp
            from detection.LPR.platerecognition.ocr import OCR
            from detection.LPR.platerecognition.utils import check_standart, get_coords, remove_duplicated_plates
            from detection.LPR.platerecognition.exceptions import OutOfBoundsError, ModelLoadError, ReadImageError, \
                WriteImageError

from keras import backend as K
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


class ALPR:
    """
    Recognition of digital licence plate names from a frame

    Attributes:

    Methods:
        detect(frame): Detect licence plate from image using image path string
    """

    def __init__(self, use_gpu=False, vehicle_threshold=0.35, plate_threshold=0.5, **kwargs):
        __vehicle_weights = dir_path + 'lpr-models/yolov3.weights'
        __vehicle_netcfg = dir_path + 'lpr-models/yolov3.cfg'
        __vehicle_dataset = dir_path + 'lpr-models/coco.data'
        __vehicle_names = dir_path + 'lpr-models/coco.names'
        __wpod_net_path = dir_path + "lpr-models/wpod-net_update1.h5"

        ##Standart thresholds specified by darknet
        self.__vehicle_threshold = vehicle_threshold
        self.__lp_threshold = plate_threshold

        self.ocr = OCR(use_gpu=use_gpu)
        self.temp_path = dir_path + "plate.png"  ## This image is temporary. Darknet need path to use Detect function.
        ## Processed image will be saved on this path and darknet will use this png image to feed ocr.

        ##Load vehicle detector model
        try:
            self.dark_alpr = Darknet(__vehicle_netcfg,
                                     __vehicle_weights,
                                     __vehicle_dataset,
                                     __vehicle_names,
                                     gpu=use_gpu)
        except Exception as e:
            raise ModelLoadError

        if use_gpu == 1:
            fraction = kwargs['fraction'] if 'fraction' in kwargs else 0.1
            config = tf.ConfigProto(
                **{'gpu_options': {'allow_growth': False, 'per_process_gpu_memory_fraction': fraction}})
            K.set_session(tf.Session(config=config))

        ##Load licence plate detector model
        try:
            self.__wpod_net = load_model(__wpod_net_path)
        except Exception as e:
            raise ModelLoadError

    def detect(self, frame):
        """
        Detect licence plate from image using image path

        Parameters:
            frame(string): Input image path

        Return:
            result(dict): {
                plate_string(list): List of digital plate name found on plate
                plate_coords(list): List of plate coordinates found in frame
                plate_standart(list): List of plate standart bool values
                plate_croped_image(numpy array): List of plate images found in frame
            }
        """

        plates_images = []
        plate_names = []
        plates_coords = []
        plate_standarts = []  ## this array holds if plate has province code, middle letters and last numbers on plate.
        ## If there is something wrong such as lack of province code or missing letter with these three standart rule then it will hold False. Otherwise true.

        ##Detect vehicle image(s) from frame
        R = self.dark_alpr.detect(frame, thresh=self.__vehicle_threshold)

        if (isinstance(frame, str)):
            try:
                frame = cv2.imread(frame)
            except Exception as e:
                raise ReadImageError

        ##if vehicle detected in frame
        if len(R):
            WH = np.array(frame.shape[1::-1], dtype=float)  ##get width height
            ##process each vehicle that found in frame
            for r in R:
                x, y, w, h = (np.array(r['bbox']) / np.concatenate((WH, WH))).tolist()
                tl = np.array([x, y])
                br = np.array([x + w, y + h])
                label = Label(0, tl, br)
                try:
                    croped_vehicle_img = crop_region(frame, label)  ##crop vehicle area from frame
                except Exception as e:
                    raise OutOfBoundsError

                ##Preprocess
                ratio = float(max(croped_vehicle_img.shape[:2])) / min(croped_vehicle_img.shape[:2])
                side = int(ratio * 288.)
                bound_dim = min(side + (side % (2 ** 4)), 608)

                ##Detect plate(s) from vehicle image(s)
                Llp, LlpImgs, what = detect_lp(self.__wpod_net,
                                               im2single(croped_vehicle_img.astype('uint8')),
                                               bound_dim,
                                               2 ** 4,
                                               (240, 80),
                                               self.__lp_threshold)

                ##if there are plate on vehicle image
                if len(LlpImgs):
                    for lp, plate in zip(Llp, LlpImgs):  ##for each plate that found in vehicles
                        plates_images.append(
                            plate)  ##add corrected image to array. Plate image is corrected in detect_lp function above
                        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                        plate = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)

                        try:
                            cv2.imwrite(self.temp_path, plate * 255.)  ##save plate to feed ocr
                        except Exception as e:
                            raise WriteImageError

                        ##Find plate coordinates on image and add them to plates_coords array
                        s = Shape(lp.pts)
                        pts = s.pts * label.wh().reshape(2, 1) + label.tl().reshape(2, 1)
                        ptspx = pts * np.array(frame.shape[1::-1], dtype=float).reshape(2, 1)
                        pt1, pt2 = get_coords(frame, ptspx)  ##coordinates of plate on frame

                        plates_coords.append([pt1, pt2])
                        plate_names.append(self.ocr.detect(self.temp_path))
                        plate_standarts.append(check_standart(plate_names[-1]))  ##check latest frame

        result = {
            "plate_string": plate_names,
            "plate_coords": plates_coords,
            "plate_standart": plate_standarts,
            "plate_croped_image": plates_images
        }
        return result

    def detect_with_alpr(self, frame):
        """
        Detect licence plate from image using image path

        Parameters:
            frame(string): Input image path

        Return:
            result(dict): {
                plate_string(list): List of digital plate name found on plate
                plate_coords(list): List of plate coordinates found in frame
                plate_standart(list): List of plate standart bool values
                plate_croped_image(numpy array): List of plate images found in frame
            }
        """

        if (isinstance(frame, str)):
            try:
                frame = cv2.imread(frame)
            except Exception as e:
                raise ReadImageError

        plates_images = []
        plate_names = []
        plates_coords = []
        plate_standarts = []
        confidence = []

        ##Detect vehicle image(s) from frame
        R = self.dark_alpr.detect(frame, thresh=self.__vehicle_threshold)

        ##if vehicle detected in frame
        if len(R):
            WH = np.array(frame.shape[1::-1], dtype=float)  ##get width height
            ##process each vehicle that found in frame
            for r in R:
                x, y, w, h = (np.array(r['bbox']) / np.concatenate((WH, WH))).tolist()
                tl = np.array([x, y])
                br = np.array([x + w, y + h])
                label = Label(0, tl, br)
                try:
                    croped_vehicle_img = crop_region(frame, label)  ##crop vehicle area from frame
                except Exception as e:
                    raise OutOfBoundsError

                ##Preprocess
                ratio = float(max(croped_vehicle_img.shape[:2])) / min(croped_vehicle_img.shape[:2])
                side = int(ratio * 288.)
                bound_dim = min(side + (side % (2 ** 4)), 608)

                ##Detect plate(s) from vehicle image(s)
                Llp, LlpImgs, _, conf = detect_lp(self.__wpod_net,
                                                  im2single(croped_vehicle_img.astype('uint8')),
                                                  bound_dim,
                                                  2 ** 4,
                                                  (240, 80),
                                                  self.__lp_threshold)

                ##if there are plate on vehicle image
                if len(LlpImgs):
                    for lp, plate, c in zip(Llp, LlpImgs, conf):  ##for each plate that found in vehicles
                        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
                        plate = (plate * 255.).astype('uint8')
                        plates_images.append(plate)
                        confidence.append(c)

                        ##Find plate coordinates on image and add them to plates_coords array
                        s = Shape(lp.pts)
                        pts = s.pts * label.wh().reshape(2, 1) + label.tl().reshape(2, 1)
                        ptspx = pts * np.array(frame.shape[1::-1], dtype=float).reshape(2, 1)
                        pt1, pt2 = get_coords(frame, ptspx)  ##coordinates of plate on frame

                        plates_coords.append([pt1, pt2])
                        plate_names.append(self.ocr.detect(plate))
                        plate_standarts.append([False, False, False])  ##check latest frame

        result = {
            "plate_strings": plate_names,
            "plate_coords": plates_coords,
            "plate_standarts": plate_standarts,
            "plate_croped_images": plates_images,
            "confidence": confidence
        }
        return result

    def detect_with_wpod(self, frame):
        if isinstance(frame, str):
            frame = cv2.imread(frame)

        ratio = float(max(frame.shape[:2])) / min(frame.shape[:2])
        side = int(ratio * 288.)
        bound_dim = min(side + (side % (2 ** 3)), 400)

        tl = np.array([0, 0])
        br = np.array([frame.shape[0], frame.shape[1]])
        label = Label(0, tl, br)

        plates_images = []
        plates_names = []
        plates_coords = []
        plate_standarts = []
        confidence = []

        Llp, LlpImgs, _, conf = detect_lp(self.__wpod_net,
                                          im2single(frame.astype('uint8')),
                                          bound_dim,
                                          2 ** 3,
                                          (240, 80),
                                          self.__lp_threshold)

        for lp, plate, c in zip(Llp, LlpImgs, conf):
            plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
            plate = (plate * 255.).astype('uint8')
            plates_images.append(plate)
            confidence.append(c)

            s = Shape(lp.pts)
            pts = s.pts * label.wh().reshape(2, 1) + label.tl().reshape(2, 1)
            ptspx = pts * np.array(frame.shape[1::-1], dtype=float).reshape(2, 1)
            pt1, pt2 = get_coords(frame, ptspx)

            # Upsample coordinates
            x, y, _ = frame.shape
            for coord in [[pt1, pt2]]:
                for i in range(4):
                    coord[0][i] = (int(coord[0][i][0] / x), int(coord[0][i][1] / y))
                    coord[1][i] = (int(coord[1][i][0] / x), int(coord[1][i][1] / y))

            plates_names.append(self.ocr.detect(plate))
            plates_coords.append([pt1, pt2])
            plate_standarts.append([False, False, False])  ##check latest frame

        result = {
            "plate_strings": plates_names,
            "plate_coords": plates_coords,
            "plate_standarts": plate_standarts,
            "plate_croped_images": plates_images,
            "confidence": confidence
        }
        return result

    def detect_v2(self, frame):
        x, y, _ = frame.shape
        xf = yf = 1
        xf = 1024 / x
        yf = 1024 / y
        resized = cv2.resize(frame, (0, 0), fx=yf, fy=xf)
        # resized = frame.copy()

        ##detections
        s = time()
        alpr_result = self.detect_with_alpr(frame)
        wpod_result = self.detect_with_wpod(resized)
        print(f"Detection finished in {time() - s} second")

        # for i in range(len(alpr_result['plate_coords'])):
        #     coord = alpr_result['plate_coords'][i]
        #     arr=[]
        #     for plate in coord:
        #         a = [(int(i/yf),int(j/xf)) for i,j in plate]
        #         arr.append(a)
        #     alpr_result['plate_coords'][i]=arr

        for i in range(len(wpod_result['plate_coords'])):
            coord = wpod_result['plate_coords'][i]
            arr = []
            for plate in coord:
                a = [(int(i / yf), int(j / xf)) for i, j in plate]
                arr.append(a)
            wpod_result['plate_coords'][i] = arr

        ##join two results
        result = {"plate_strings": alpr_result['plate_strings'] + wpod_result['plate_strings'],
                  "plate_coords": alpr_result['plate_coords'] + wpod_result['plate_coords'],
                  "plate_standarts": alpr_result['plate_standarts'] + wpod_result['plate_standarts'],
                  "plate_croped_images": alpr_result['plate_croped_images'] + wpod_result['plate_croped_images'],
                  "confidence": alpr_result['confidence'] + wpod_result['confidence']}

        return result

    def perform_ocr(self, frame):
        return self.ocr.detect(frame)


if __name__ == "__main__":

    import platerecognition as pr
    import os

    lpr = pr.LicencePlateRecognition()

    anno = os.listdir("pathanno")
    anno = [i.split(".")[0] for i in anno]
    images = sorted(["pathimages/" + i + ".jpg" for i in anno])
    anno = sorted(["pathanno/" + i + ".txt" for i in anno])

    frames = [cv2.imread(file) for file in images]
    strings = []
    for i in anno:
        with open(i, "r") as file:
            s = file.readline().split("	")[-1].rstrip()
            strings.append(s)

    i = 0
    images = []
    plates = []
    correct = 0
    wrong = 0
    for frame in frames:
        result = lpr.model.detect_v2(frame)
        result['plate_coords'], result['plate_strings'], result['plate_standarts'], result[
            'plate_croped_images'] = remove_duplicated_plates(result['plate_coords'], result['plate_strings'],
                                                              result['plate_standarts'], result['plate_croped_images'])
        cpy = frame.copy()
        draw(cpy, result['plate_coords'])
        cv2.imwrite("path/benchmark_results/" + str(i + 1) + ".jpg", cpy)
        w = True
        print("_" * 50 + "\n")
        print(result['plate_strings'][0], strings[i])
        for predicted_str in result['plate_strings']:
            if predicted_str == strings[i]:
                correct += 1
                w = False
        if w == True:
            wrong += 1
        i += 1

        print(f"Accuracy: {correct / i}, Correct: {correct}, Wrong: {wrong}, Total processed: {i}")
