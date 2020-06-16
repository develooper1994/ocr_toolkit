import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
import cv2

import torch
from torchvision.transforms import transforms

from imageio import imread
from PIL import Image, ImageOps
import timm

alphabets_dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
                 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
                 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}
# alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
#              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
alphabets = list(alphabets_dic.values())
alphabets = [*alphabets[26:], *alphabets[:26]]

l = open("labels.pickle", "rb")
labels = pickle.load(l)

current_path = os.getcwd()
model_path = current_path + "/efficientnet_b3a/efficientnet_b3a_9922.pth.tar"
model = timm.create_model("efficientnet_b3a", pretrained=False, num_classes=36,
                          checkpoint_path=model_path)
model.eval()

plate_number = "4"
plate_crop = "crop_" + plate_number + ".png"
plate_name = "01DJP58" + ".JPG"
plate_path = "plate_output" + "/"
plate_full_path = plate_path + plate_name + "/" + plate_crop
# plate_full_path = r"C:\Users\selcu\PycharmProjects\ocr_toolkit\recognition\licence_plate\ANPR-master\Licence_plate_recognition\USA_plates\dataset\0/0_1.jpg"
digit_image = Image.open(plate_full_path)
# digit_image.convert("")

digit_image_arr = np.array(digit_image)


def BINARY_INV(image_numpy):
    _, digit_image_arr = cv2.threshold(image_numpy, 127, 255, cv2.THRESH_BINARY_INV)
    return digit_image_arr


# plt.imshow(th, cmap="gray")
# plt.show()

# digit_image_arr_expanded = np.expand_dims(digit_image_arr, axis=0)
# digit_image_arr_expanded = np.tile(digit_image_arr_expanded, [3, 1, 1])
# digit_image_torch = torch.from_numpy(digit_image_arr_expanded)
# digit_image_torch_unsqueeze = digit_image_torch.unsqueeze(0).float()
# digit_image = cv2.resize(digit_image, new_size, cv2.INTER_CUBIC)

# %% transform
mean = np.mean(digit_image_arr, axis=(0, 1))
std = np.std(digit_image_arr, axis=(0, 1))
new_size = (28, 28)
color_inverse = lambda image: ImageOps.invert(image)  # inverted color
transform = transforms.Compose([
    transforms.Lambda(BINARY_INV),
    transforms.ToPILImage(),
    # transforms.Lambda(color_inverse),
    transforms.Resize(new_size, interpolation=transforms.Image.BICUBIC),
    transforms.ToTensor(),
    # transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
])

image_tensor = transform(digit_image_arr).float()
image_tensor = image_tensor.unsqueeze(0)
# plt.imshow(image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy())
# plt.show()

# %% get prediction
y = model(image_tensor)  # image_tensor.shape = torch.Size([1, 3, 28, 28])

calc_confidence = torch.nn.functional.softmax

conf = calc_confidence(y)
top5_val, top5_idx = conf.topk(5)
top5_idx = top5_idx.cpu().numpy()
for idx, conf_ in zip(top5_idx[0], top5_val[0]):
    print(f"{alphabets[idx]}: {conf_ * 100:2f}%")
