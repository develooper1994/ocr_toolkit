import os
from os import path
from os.path import splitext, basename
import json
from pprint import pprint


## find plates with paths, names and characters
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


if __name__ == "__main__":
    plates_with_characters = dumps_dataset()
    ## open a text file and save plates with paths, names and characters.
    log_name = "dataset_log.txt"

    plates_with_characters_dump = json.dumps(plates_with_characters)
    with open(log_name, "w") as logfile:
        logfile.write(plates_with_characters_dump)
