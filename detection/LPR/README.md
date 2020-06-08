# Licence Plate Recognition

## Installation
* pip install --index-url http://pypi.turkai.com platerecognition --trusted-host pypi.turkai.com
* pip install --index-url http://pypi.turkai.com darknet --trusted-host pypi.turkai.com

**For GPU**
* pip install tensorflow-gpu==1.14.0

## Usage
```
import cv2
path="" ##fill here

import platerecognition as pr
gpu = False
lpr = pr.LicencePlateRecognition(use_gpu=gpu)

plate = lpr.detect(path)['plate_string']
```

