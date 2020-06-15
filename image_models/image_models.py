import os

import numpy as np

import torch
from torchvision.transforms import transforms

from imageio import imread
import timm

alphabets_dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
                 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
                 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}
alphabets = alphabets_dic.values()
dataset_classes = [cls for cls in alphabets]

current_path = os.getcwd()
model_path = current_path+"/efficientnet_b3a/efficientnet_b3a_9922.pth.tar"
model = timm.create_model("efficientnet_b3a", pretrained=False,
                          checkpoint_path=model_path)
model.eval()

plate_crop = "crop_1" + ".png"
plate_name = "01DJP58" + ".JPG"
plate_path = "plate_output" + "/"
plate_full_path = plate_path + plate_name + "/" + plate_crop
digit_image = imread(plate_full_path)

mean = np.mean(digit_image, axis=(0, 1))
std = np.std(digit_image, axis=(0, 1))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
)

image_tensor = transform(digit_image).float()
image_tensor = image_tensor.unsqueeze(0)

y = model(image_tensor)

calc_confidence = torch.nn.functional.softmax

conf = calc_confidence(y)
top5_val, top5_idx = conf.topk(5)
top5_idx = top5_idx.cpu().numpy()
# for idx, conf_ in zip(top5_idx[0], top5_val[0]):
#     print('{}: {:2f}%'.format(labels.cls_idx[idx], conf_ * 100))
