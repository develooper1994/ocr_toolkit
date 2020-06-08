import sys

import cv2
import pandas as pd

try:
    import ocr as OCR
except:
    try:
        import platerecognition.ocr as OCR
    except:
        try:
            import LPR.platerecognition.ocr as OCR
        except:
            import detection.LPR.platerecognition.ocr as OCR

try:
    use_gpu = sys.argv[1]
except Exception as e:
    use_gpu = "0"

if use_gpu == "1":
    ocr_model = OCR(use_gpu=True)
else:
    ocr_model = OCR()

data = pd.read_csv("data/trainVal.csv")
data = data.values

correct = 0
wrong = 0
counter = 0
import time

for sample in data:
    start_time = time.time()
    grand_image = cv2.imread("data/" + sample[1])
    io_time = time.time()
    grand_string = sample[2]
    predicted_string = ocr_model.detect(grand_image)
    ocr_time = time.time()

    if predicted_string == grand_string:
        correct += 1
    else:
        wrong += 1

    counter += 1
    print(f"Accuracy:{correct / counter} - Correct: {correct} - Wrong: {wrong}")
    print(
        f"ocr inference time: {ocr_time - io_time} io time: {io_time - start_time} total time: {ocr_time - start_time}")

print(f"Accuracy:{correct / counter}")
