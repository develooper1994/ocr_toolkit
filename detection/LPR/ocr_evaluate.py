import os
import cv2

from .platerecognition.alpr import ALPR

lpr_model = ALPR(use_gpu=False, fraction=0.5)
path = "../tr_plaka/"  ##burası dolacak
files = os.listdir(path)
size = len(files)
true = 0
for file in files:
    text = file.split(".")[0]
    image = cv2.imread(path + file)
    result = lpr_model.perform_ocr(image)

    if text == result:
        true += 1

print(true / size)
