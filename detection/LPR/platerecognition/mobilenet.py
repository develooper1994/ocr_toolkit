import sys
from os import path

dir_path = path.dirname(path.abspath(__file__)) + "/"
sys.path.append(dir_path)
import cv2

try:
    import OCR
    import check_standart
    import ModelLoadError, ReadImageError, WriteImageError
except:
    try:
        from platerecognition.ocr import OCR
        from platerecognition.utils import check_standart
        from platerecognition.exceptions import ModelLoadError, ReadImageError, WriteImageError
    except:
        try:
            from LPR.platerecognition.ocr import OCR
            from LPR.platerecognition.utils import check_standart
            from LPR.platerecognition.exceptions import ModelLoadError, ReadImageError, WriteImageError
        except:
            from detection.LPR.platerecognition.ocr import OCR
            from detection.LPR.platerecognition.utils import check_standart
            from detection.LPR.platerecognition.exceptions import ModelLoadError, ReadImageError, WriteImageError


class MobileNet:
    """
    Mobilenet and SSD model module

    This module is using mobilenet-ssd model to find licence plate in a frame

    Attributes:

    Methods:
        detect(frame): Takes a frame and detect licence plate from frame using mobilenet model

    NOTE: Mobilenet use opencv plate detection algorithm that called wpod-net to find plates. Unlike alpr, mobilenet cannot do affine transformation on plates.
    It sends untransformed original plates to ocr. Therefore, ocr results may not be as good as alpr so alpr does affine transformation before sending plate images to ocr.
    """

    def __init__(self):
        try:
            self.__model = cv2.dnn.readNetFromCaffe(dir_path + "lpr-models/MobileNetSSD_test.prototxt",
                                                    dir_path + "lpr-models/lpr.caffemodel")
        except Exception as e:
            # print("[ERROR] Loading Mobilenet LPR caffe model has failed: " + str(e))
            # return
            raise ModelLoadError
        print("[INFO] Mobilenet LPR caffe model succesfully loaded...")

        self.__inWidth = 720
        self.__inHeight = 1024
        self.__inScaleFactor = 0.007843
        self.__meanVal = 127.5

        self.__ocr = OCR()
        self.__temp_path = dir_path + "plate.png"  ## This image is temporary. Darknet need path to use Detect function.
        ## Processed image will be saved on this path and darknet will use this png image to feed ocr.

    def detect(self, frame):
        ''' Detecting licence plate from image

        Parameters:
            frame(numpy array): Frame to detect licence plate from

        Return:
            result(dict): {
                plate_string(list): List of digital plate name found on plate
                plate_coords(list): List of plate coordinates found in frame
                plate_standart(list): List of plate standart bool values
                plate_croped_image(numpy array): List of plate images found in frame
            }
        '''
        if (isinstance(frame, str)):
            try:
                frame = cv2.imread(frame)
            except Exception as e:
                raise ReadImageError

        ##return valeus
        plates_images = []
        plates_names = []
        plates_coords = []
        plates_standarts = []

        blob = cv2.dnn.blobFromImage(frame, self.__inScaleFactor, (self.__inWidth, self.__inHeight), self.__meanVal)
        self.__model.setInput(blob)
        detections = self.__model.forward()
        cols = frame.shape[1]
        rows = frame.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                plates_coords.append([yLeftBottom, yRightTop, xLeftBottom, xRightTop])
                image_sub = frame[yLeftBottom:yRightTop, xLeftBottom:xRightTop]

                ## TODO: Affine transformation
                # # # # # # # # # # # # # # # # # # # # # # # # # # # 3
                # pts1 = np.float32([[0,0],[60,3],[0,34],[60,31]])
                # pts2 = np.float32([[0,0],[100,0],[0,100],[100,100]])
                # M = cv2.getPerspectiveTransform(pts1,pts2)
                # cv2.imshow("asd", image_sub)
                # dst = cv2.warpPerspective(image_sub,M,(200,200))
                # cv2.imshow("asd2", dst)
                # cv2.waitKey(0)

                # x,y = image_sub.shape[1], image_sub.shape[0]
                # t_ptsh 	= np.matrix([[0,x,x,0],[0,0,y,y],[1.,1.,1.,1.]],dtype=float)
                # ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
                # H 		= find_T_matrix(ptsh,t_ptsh)
                # Ilp 	= cv2.warpPerspective(image_sub,H,(y,x),borderValue=.0)
                # # # # # # # # # # # # # # # # # # # # # # # # # # # 3
                try:
                    cv2.imwrite(self.__temp_path, image_sub)  ##write image to file so that ocr can use it as img path
                except Exception as e:
                    raise WriteImageError
                lp_str = self.__ocr.detect(self.__temp_path)

                plates_names.append(lp_str)
                plates_images.append(image_sub)
                plates_standarts.append(check_standart(plates_names[-1]))

        result = {
            "plate_strings": plates_names,
            "plate_coords": plates_coords,
            "plate_standart": plates_standarts,
            "plate_croped_images": plates_images
        }
        return result
