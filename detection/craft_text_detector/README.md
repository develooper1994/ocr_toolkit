https://pypi.org/project/craft-text-detector/
# CRAFT: Character-Region Awareness For Text detection

Packaged, Pytorch-based, easy to use, cross-platform version of the CRAFT text detector | 
[Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)

**[Youngmin Baek](mailto:youngmin.baek@navercorp.com), Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee.**
 
## Overview

PyTorch implementation for CRAFT text detector that effectively detect text area by exploring 
each character region and affinity between characters. The bounding box of texts are obtained 
by simply finding minimum bounding rectangles on binary map after thresholding character region 
and affinity scores.

<img width="1172" alt="teaser" src="./figures/craft_example.gif">

## Getting started
### Install dependencies
#### Requirements
    torch>=0.4.1
    torchvision>=0.2.1
    opencv-python>=3.4.8.29
    scipy>=1.3.2
    gdown>=3.10.1
    setuptools>=46.2.0
    numpy>=1.18.1
    argparse>=1.4.0
    pillow>=7.0.0
    scikit-image>=0.17.1
    h5py>=2.10.0
- check requiremtns.txt
```
pip install -r requirements.txt
```

#### Install using pip for Linux and Mac:

```console
pip install git+https://github.com/develooper1994/craft_text_detector
```
or
```download and install
python setup.py install
```

### Basic Usage

```python
# import package
import craft_text_detector as craft

# set image path and export folder directory
image_path = 'figures/idcard.png'
output_dir = 'outputs/'

# apply craft text detection and export detected regions to output directory
prediction_result = craft.detect_text(image_path,output_dir, device="cpu", crop_type="is_poly")
```

### Advanced Usage

```python
# import package
import craft_text_detector as craft

# set image path and export folder directory
image_name = 'idcard.png'
image_path = 'figures/' + image_name
output_dir = 'outputs/'

# read image
image = craft.imgproc.read_image(,

# load models
craft_model_path = "craft_mlt_25k.pth"
refinenet_model_path = "craft_refiner_CTW1500.pth"
pred = craft.craft_detector.craft_detector(image=image,
                                           craft_model_path=craft_model_path,
                                           refinenet_model_path=refinenet_model_path,
                                           device="gpu")

# perform prediction
text_threshold = 0.9
link_threshold = 0.2
low_text = 0.2
device="gpu"  # False
show_time = False
# perform prediction
prediction_result = pred.get_prediction(image=image,text_threshold=0.7,link_threshold=0.4,low_text=0.4,square_size=1280,show_time=True)
# export detected text regions
exported_file_paths = craft.file_utils.export_detected_regions(
    image_path=image_path,
    image=image,
    regions=prediction_result["boxes"],
    output_dir=output_dir,
    rectify=True
)
# export heatmap, detection points, box visualization
craft.file_utils.export_extra_results(
    image_path=image_path,
    image=image,
    regions=prediction_result["boxes"],
    heatmaps=prediction_result["heatmaps"],
    output_dir=output_dir
)
```

### Test instruction using pretrained model
- Download the trained models from my Google Drive
 
 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link Baidu* | *Model Link Google Drive* |
 | :--- | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose |  [SynthText+IC13+IC17 for baidu drive](https://pan.baidu.com/s/1PTTzbM9XG0pNe5i-uL6Aag) |      [SynthText+IC13+IC17 for google drive](https://drive.google.com/open?id=1CV3Ao4zuDikOkuHHNCw0_YRVWq7YsPBQ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [SynthText+IC15 for baidu drive](https://pan.baidu.com/s/19lJRM6YWZXVkZ_aytsYSiQ) |     [SynthText+IC15 for google drive](https://drive.google.com/open?id=1zQYaWF9_9Jsu5xjA5tD0X9N6Ug0lnbtm)
SynthText | SynthText | - | For SynthText only | [SynthText for baidu drive](https://pan.baidu.com/s/1MaznjE79JNS9Ld48ZtRefg) |     [SynthText for google drive](https://drive.google.com/open?id=1pzPBZ5cYDCHPVRYbWTgIjhntA_LLSLyS)                                                                                                    
LinkRefiner | CTW1500 | - | Used with the General Model |  [LinkRefiner for baidu drive]() |     [LinkRefiner for google drive](https://drive.google.com/open?id=1ZDe0WRwlxLRkwofQt8C18ZTF-oA3vbfs)

* Run with pretrained model
``` (with upto python 3.7)
python test_on_dataset.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```

The result image and score maps will be saved to `./result` by default.

### Arguments
* `--trained_model`: pretrained model
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--device`: use device for inference (default:"gpu")
* `--canvas_size`: max image size for inference
* `--mag_ratio`: image magnification ratio
* `--poly`: enable polygon type result
* `--show_time`: show processing time
* `--test_folder`: folder path to input images
* `--refine`: use link refiner for sentense-level dataset
* `--refiner_model`: pretrained refiner model

### Training
Planning!!! <br>
`Note: When you train the IC15-Data or MLT-Data, please see the annotation in data_loader.py line 92 and line 108-112.`







## Links
- WebDemo : https://demo.ocr.clova.ai/

## Citation
```
@inproceedings{baek2019character,
  title={Character Region Awareness for Text Detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}
```