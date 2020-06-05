# %% modules

from pprint import pprint

import craft_text_detector as craft
import numpy as np

# %% md

#### set image path and export folder directory

# %%

image_name = 'idcard.png'
image_path = 'figures/' + image_name
output_dir = 'outputs/'

# %% md

#### read image

# %%

image = craft.imgproc.read_image(image_path)

# %% md

#### load models

# %%

craft_model_path = "../craft_mlt_25k.pth"
refinenet_model_path = "../craft_refiner_CTW1500.pth"
craft_net = craft.craft_detector.craft_detector(image=image,
                                                craft_model_path=craft_model_path,
                                                refinenet_model_path=refinenet_model_path,
                                                cuda=True)

# %% md

#### perform prediction

# %%

text_threshold = 0.9
link_threshold = 0.2
low_text = 0.2
cuda = True  # False
show_time = False
# perform prediction
prediction_result = craft_net(image=image,
                              text_threshold=0.7,
                              link_threshold=0.4,
                              low_text=0.4,
                              square_size=1280,
                              show_time=True)

# %% md

### Inspect predicted results

# %%

keys = prediction_result.keys()
heatmap_keys = prediction_result["heatmaps"].keys()
pprint(keys)
pprint(heatmap_keys)

# %%

boxes = prediction_result['boxes']
boxes_as_ratios = prediction_result['boxes_as_ratios']
polys = prediction_result['polys']
polys_as_ratios = prediction_result['polys_as_ratios']
text_score_heatmap = prediction_result['heatmaps']['text_score_heatmap']
link_score_heatmap = prediction_result['heatmaps']['link_score_heatmap']
pprint(boxes.shape)
pprint(boxes_as_ratios.shape)
pprint(polys.shape)
pprint(polys_as_ratios.shape)
pprint(text_score_heatmap.shape)
pprint(link_score_heatmap.shape)

# %%

coords = np.array(boxes).astype(np.int32)
pprint(coords.shape)
# pprint(coords[0:5])

# 0---1
# |   |
# 3---2
upper_left = coords[:, 0]
upper_left_x = upper_left[:, 0]
upper_left_x_arg = np.argsort(upper_left_x)
upper_left_y = upper_left[:, 1]
upper_left_y_arg = np.argsort(upper_left_y)  # horizontal line


# note necessary
# bottom_right = coords[:, 2]
# bottom_right_x = coords[:, 0]
# bottom_right_x_arg = np.argsort(bottom_right_x)
# bottom_right_y = coords[:, 1]
# bottom_right_y_arg = np.argsort(bottom_right_y)

# line_bbs = sort_bbs_line_by_line(boxes)

def coord_sort(boxes):
    line = []
    lines = []
    number_of_box = boxes.shape[0]
    # all bounding boxes.
    for i, box in enumerate(boxes):
        previous_box = None
        box_x0, box_y0 = box[0::2]  # top- left corner <-> bottom- right corner
        if previous_box is not None:
            pass


# coord_sort(boxes)

# %% md

#### export detected text regions

# %%

exported_file_paths = craft.file_utils.export_detected_regions(
    image_path=image_path,
    image=image,
    regions=prediction_result["boxes"],
    output_dir=output_dir,
    rectify=True
)

# %% md

#### export heatmap, detection points, box visualization

# %%

# gives (image)_text_detection.txt file
craft.file_utils.export_extra_results(
    image_path=image_path,
    image=image,
    regions=prediction_result["boxes"],
    heatmaps=prediction_result["heatmaps"],
    output_dir=output_dir
)
