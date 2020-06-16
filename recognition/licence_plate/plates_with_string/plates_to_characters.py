import os
from os import path
from os.path import splitext, basename
import json
from pprint import pprint

## find plates with paths, names and characters
from detection.detection_selector import detection_selector


def only_plate(plate_name_with_extention):
    plate, extention = splitext(basename(plate_name_with_extention))
    return plate


def split_plate_characters(plate_name):
    characters = []
    for c in plate_name:
        characters.append(c)
    return characters


def dumps_dataset(dataset_path=None):
    if dataset_path is None:
        dataset_path = "dataset"
    current_path = os.getcwd()
    dataset_path = os.path.join(current_path, dataset_path)
    plates_with_characters = {}
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            root_name = os.path.join(root, name)
            # print(root_name)

            plate_name = only_plate(name)
            characters = split_plate_characters(plate_name)
            plates_with_characters[name] = [root_name, characters]
    # pprint(plates_with_characters)
    return plates_with_characters


def loads_dataset(plates_with_characters_object):
    plate_names = list(plates_with_characters_object.keys())
    plates_with_characters_object_values = list(plates_with_characters_object.values())
    root_names, characterss = [], []
    for plates_with_characters in plates_with_characters_object_values:
        root_names.append(plates_with_characters[0])
        characterss.append(plates_with_characters[1])

    return plate_names, root_names, characterss


def plate_to_character_images(dev_selector, image_path, only_characters=True, show_time=False, crop_type="box"):
    prediction_result = dev_selector.detect_text(image=image_path, output_dir=None, rectify=True,
                                                 export_extra=True,
                                                 text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                                                 only_characters=only_characters, square_size=720,
                                                 show_time=show_time, crop_type=crop_type)
    character_images = dev_selector.get_detected_polygons()
    return character_images


def save_character_images(character_images):
    raise NotImplementedError


def plates_image_to_characters(dev_selector, plates_with_characters_object, show_time=False, crop_type="box"):
    plate_names, root_names, characterss = loads_dataset(plates_with_characters_object)
    for plate_name, root_name, characters in zip(plate_names, root_names, characterss):
        character_images = plate_to_character_images(dev_selector, root_name)
        save_character_images(character_images)


if __name__ == "__main__":
    plates_with_characters = dumps_dataset()
    ## open a text file and save plates with paths, names and characters.
    log_name = "dataset_log.txt"

    plates_with_characters_dump = json.dumps(plates_with_characters)
    with open(log_name, "w") as logfile:
        logfile.write(plates_with_characters_dump)

    ## detection
    selector = "craft"
    detection_model_paths = ["craft_mlt_25k.pth", "craft_refiner_CTW1500.pth"]
    dev_selector = detection_selector(None, detection_switch=selector, device="gpu")
    plates_with_characters_object = json.loads(plates_with_characters_dump)
    plates_image_to_characters(dev_selector, plates_with_characters_object)
