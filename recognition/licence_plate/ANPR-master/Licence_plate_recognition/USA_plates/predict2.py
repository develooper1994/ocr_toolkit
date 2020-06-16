import cv2
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from imageio import imread

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

try:
    from digits import plate_segmentation
except:
    from .digits import plate_segmentation

# Load model
model = load_model('cnn_classifier.h5')

# Detect chars
plate_number = "1"
plate_crop = "crop_" + plate_number + ".png"
plate_name = "01DJP58" + ".JPG"
plate_path = "plate_output" + "/"
plate_full_path = plate_path + plate_name + "/" + plate_crop
# plate_full_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\recognition\licence_plate\ANPR-master\Licence_plate_recognition\USA_plates\dataset\0/0_1.jpg"
digit_image = imread(plate_full_path)
digit_image = cv2.cvtColor(digit_image, cv2.COLOR_RGB2GRAY)
digit_image = cv2.resize(digit_image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
_, digit_image_arr = cv2.threshold(digit_image, 127, 255, cv2.THRESH_BINARY_INV)

digits = [digit_image]
# Predict
for d in digits:
    d = np.reshape(d, (1, 28, 28, 1))
    out = model.predict(d)
    # Get max pre arg
    p = []
    precision = 0
    for i, output in enumerate(out):
        z = np.zeros(36)
        z[np.argmax(output)] = 1.
        precision = max(out[i])
        p.append(z)
    prediction = np.array(p)

    # Inverse one hot encoding
    alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    classes = []
    for a in alphabets:
        classes.append([a])
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(classes)
    pred = ohe.inverse_transform(prediction)

    print('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))
