import os
import random
from ctypes import *

import cv2 as cv
import numpy as np


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class Darknet:
    """Darknet class to perform detections with Darknet framework.

    Attributes:
        net: Detection network.
        meta: Detection meta.
    
    Methods:
        detect(img_path, thresh, hier_thresh, nms): Performs detection.
        visualize(img_path, dets, thresh): Draws and visualizes the boundary boxes of the detected objects.
        init_colors(names): Generates a unique color for each class.
        array_to_image (arr): Converts given image array to Image object.
    """

    def __init__(self, cfg, weights, data, names, gpu=None):
        """Loads and initializes net and meta.

        Parameters:
            cfg (str): Full path to '.cfg' file.
            weights (str): Full path to '.weights' file.   
            data (str): Full path to '.data' file.
            names (str): Full path to '.names' file.
            gpu (bool): If True, GPU is used.
                (default is None)
        """

        if gpu is None:
            gpu = os.system("nvcc --version") == 0

        self.load_funtions(gpu)

        with open(data, 'r') as file:
            content = file.readlines()
        for i in range(len(content)):
            content[i] = content[i].strip()
            if content[i].startswith('names'):
                content[i] = 'names = ' + names
        with open(data, 'w') as file:
            for i in content:
                file.write("{}\n".format(i))

        self.colors = self.init_colors(names)

        self.net = self.load_net_custom(cfg.encode('utf-8'), weights.encode('utf-8'), 0, 1)
        self.meta = self.load_meta(data.encode('utf-8'))

    def detect(self, img, thresh=.5, hier_thresh=.5, nms=.45):
        """Performs detection.

        Reads the image from the given path and performs detection.
        Creates a list containing information about the objects detected with probability (confidence)
        above the given threshold and returns it.
        Returns None if image could not be loaded from the given path.

        Parameters:
            img (str | rgb numpy array): image to be processed.
            thresh (float): Minimum threshold value for detection. Objects detected with less probability (confidence)
                are ignored. (default is 0.5)
            
        Returns:
            dets (list): List of detected objects.

        Raises:
            Exception: Could not load image from the given path.
        """

        if type(img) is type(None):
            raise Exception("Could not load image from:", img)
        is_image_free = True
        if type(img) == str:
            im = self.load_image(img.encode('utf-8'), 0, 0)
        else:
            is_image_free = False
            im, _ = self.array_to_image(cv.cvtColor(img, cv.COLOR_RGB2BGR))

        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, im)
        letter_box = 0
        raw_det = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
        num = pnum[0]
        if nms:
            self.do_nms_obj(raw_det, num, self.meta.classes, nms)

        dets = []
        for j in range(num):
            for i in range(self.meta.classes):
                if raw_det[j].prob[i] > 0:
                    b = raw_det[j].bbox
                    temp = {}
                    temp['label'] = self.meta.names[i].decode('utf-8')
                    temp['perc'] = raw_det[j].prob[i]
                    temp['bbox'] = (b.x - b.w / 2, b.y - b.h / 2, b.w, b.h)
                    dets.append(temp)
        if is_image_free:
            self.free_image(im)
        self.free_detections(raw_det, num)

        return dets

    def visualize(self, img, dets=None, thresh=0.5, show=False):
        """Visualizes the boundary boxes of the detected objects.

        Draws and visualizes the boundary boxes of the detected objects.
        If dets is None, performs detection first.
        Returns None if image could not be loaded from the given path.

        Parameters:
            img (str | bgr np.ndarray): image to be visualized
            dets (list): List of detections to be visualized on the given image.
                (default is None)
            thresh (float): Minimum threshold value for detection. Objects detected with less probability (confidence)
                are ignored. (default is 0.5)
            show (boolean): Displays the image if set True.
                (default is False)
        
        Returns:
            img (numpy.ndarray): Image with the bboxes drawn on it.
        
        Raises:
            Exception: Could not load image from the given path.
        """

        if dets is None:
            dets = self.detect(img, thresh=thresh)

        if type(img) == str:
            img = cv.imread(img)

        if dets is None or img is None:
            raise Exception("Could not load image from:", img)

        # Start drawing for each object
        for i in dets:
            label = i['label']
            perc = format(i['perc'], '.2f')
            box_label = label + " " + perc
            x, y, w, h = i['bbox']
            top = int(y - h / 2)
            left = int(x - w / 2)
            right = int(x + w / 2)
            bottom = int(y + h / 2)

            # Prepare parameters for drawing
            color = self.colors[label]
            font = cv.FONT_HERSHEY_PLAIN
            font_scale = 1
            line_type = cv.FILLED
            text_thickness = 1

            # Draw bounding box
            cv.rectangle(img, (left, top), (right, bottom), color, 2)
            # Get label text size
            text_size, pad = cv.getTextSize(box_label, font, font_scale + 0.1, text_thickness)
            # Draw bounding box for label
            cv.rectangle(img, (left, top - text_size[1] - pad), (left + text_size[0], top), color, line_type)
            # Put label text
            cv.putText(img, box_label, (left, top - int(pad / 2)), font, font_scale, (0, 0, 0), text_thickness,
                       line_type)

        if show:
            cv.imshow('img', img)
            cv.waitKey(0)

        return img

    def init_colors(self, names):
        """Initializes colors

        Assigns a unique color for each class in the '.names' file.
        Returns None if '.names' file is empty.

        Parameters:
            names (str): Full path to '.names' file where a list of classes is stored.

        Returns:
            colors (dict): A dictionary mapping each class to a unique color.

        Raises:
            Exception: '.names' file is empty.
        """

        colors = {}
        with open(names, 'r') as f:
            classes = f.read().split('\n')[:-1]  # Ignoring empty string at the end of the list

        if len(classes) == 0:
            raise Exception("'.names' file is empty.")

        step = 256 / len(classes)
        for i in range(len(classes)):
            b = int((random.random() * 256 + i * step) % 256)
            g = int((random.random() * 256 + i * step) % 256)
            r = int((random.random() * 256 + i * step) % 256)
            colors[classes[i]] = (b, g, r)

        return colors

    def array_to_image(self, arr):
        """Converts an image array to Darknet readable format.

        Parameters:
            arr (numpy.array): Image array to be converted.
        
        Returns:
            im: Darknet readable image.
            arr (numpy.array): Image array. Needs to be returned to avoid python freeing memory.
        """

        # need to return old values to avoid python freeing memory
        arr = arr.transpose(2, 0, 1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(w, h, c, data)
        return im, arr

    def load_funtions(self, gpu):
        darknet_path = os.path.dirname(os.path.abspath(__file__))
        if gpu:
            lib = CDLL(os.path.join(darknet_path, "gpu/libdarknet.so"), RTLD_GLOBAL)
            set_gpu = lib.cuda_set_device
            set_gpu.argtypes = [c_int]
        else:
            lib = CDLL(os.path.join(darknet_path, "cpu/libdarknet.so"), RTLD_GLOBAL)

        self.get_network_boxes = lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int,
                                           POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_image = lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.load_net_custom = lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.load_meta = lib.get_metadata
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = METADATA

        self.load_image = lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.do_nms_obj = lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.predict_image = lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)
