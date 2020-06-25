## link: http://www.inf.ufpr.br/vri/databases/UFPR-ALPR.zip

import os
import string
from os import walk, path
from os.path import splitext, basename
from path import Path

import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader


class UFPR_ALPR_dataset(Dataset):
    def __init__(self, data_path=None, dataset_type="train", crop_plate=True, index_maximum=None, transform=None, download=False):
        if data_path is None:
            data_path = os.getcwd()
        self.data_path = data_path
        dataset_type = dataset_type.lower()
        if dataset_type == "train":
            self.dataset_type = "training"
        elif dataset_type == "valid" or dataset_type == "dev":
            self.dataset_type = "validtion"
        elif dataset_type == "test":
            self.dataset_type = "testing"
        else:
            self.dataset_type = ""
        self.index_maximum = index_maximum
        self.transform = transform
        # if download:
        #     download()
        # metadata
        self.parsed_metadatas = []

        self.images = []
        self.image_paths = []
        self.image_metadata_txt_paths = []
        self.image_metadata_xml_paths = []

        self.image_paths, self.image_metadata_txt_paths, self.image_metadata_xml_paths = self.get_file_paths()
        assert not self.image_paths == [] or self.image_metadata_txt_paths == [] or self.image_metadata_xml_paths == [], \
            "Please download from http://www.inf.ufpr.br/vri/databases/UFPR-ALPR.zip"

        if index_maximum is None:
            self.index_maximum = len(self.image_paths)

        self.parsed_metadatas = self.parse_metadata_txt()
        self.images = self.read_images(crop_plate=crop_plate)

        ## decode string
        # pool = string.printable
        pool = ''
        pool += string.ascii_letters
        pool += "0123456789"
        pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
        pool += ' '
        self.keys = list(pool)
        self.values = np.array(range(1, len(pool) + 1))
        self.encode_dict = dict(zip(self.keys, self.values))

        self.decode_dict = dict((v, k) for k, v in self.encode_dict.items())
        self.decode_dict.update({93: "OOK"})

    def encode(self, strings):
        # strings = [" ".join(take_10(word_tokenize(x))) for x in strings]
        strings_ = [list(x) for x in strings]
        self.strings_int = [[self.encode_dict[x.lower()] if x.lower() in self.keys else 67 for x in m] for m in strings_]
        self.strings_int = np.array(self.strings_int)
        return self.strings_int

    def decode(self, max_target):
        decoded = [self.decode_dict[x] for x in max_target.tolist()]
        return decoded

    def __getitem__(self, item):
        # assert not item > len(self.images), "!!! index access exceeded !!!"
        images = self.images[item]
        if self.transform is not None:
            images = self.transform(images)

        parsed_metadatas = self.parsed_metadatas[item]
        if isinstance(item, int):
            plate = [parsed_metadatas["plate"]]
            # position = [parsed_metadatas["position_plate"]]
        else:
            plate = [parsed_metadata["plate"] for parsed_metadata in parsed_metadatas]
            # position = [parsed_metadata["position_plate"] for parsed_metadata in parsed_metadatas]
        # return image, plate, position
        plate_encoded = self.encode(plate)[0]
        plate_encoded_len = len(plate_encoded)
        images_len = images.shape[2]
        return images, plate_encoded, images_len, plate_encoded_len

    def __reversed__(self):
        return self[::-1]

    def __len__(self):
        return len(self.images)

    # TODO! download and extract ULPR_ALPR dataset
    # @property
    # def processed_folder(self):
    #     """@TODO: Docs. Contribution is welcome."""
    #     return os.path.join(self.data_path, self.__class__.__name__, "processed")
    #
    # def _check_exists(self):
    #     return os.path.exists(
    #         os.path.join(self.processed_folder, "training")
    #     ) and os.path.exists(
    #         os.path.join(self.processed_folder, "validation")
    #     ) and os.path.exists(
    #         os.path.join(self.processed_folder, "testing")
    #     )
    #
    # def download(self):
    #     """Download the MNIST data if it doesn't exist in processed_folder."""
    #     if self._check_exists():
    #         return
    #
    #     os.makedirs(self.raw_folder, exist_ok=True)
    #     os.makedirs(self.processed_folder, exist_ok=True)
    #
    #     # download files
    #     for url, md5 in self.resources:
    #         filename = url.rpartition("/")[2]
    #         download_and_extract_archive(
    #             url, download_root=self.raw_folder, filename=filename, md5=md5
    #         )
    #
    #     # process and save as torch files
    #     print("Processing...")
    #
    #     training_set = (
    #         read_image_file(
    #             os.path.join(self.raw_folder, "train-images-idx3-ubyte")
    #         ),
    #         read_label_file(
    #             os.path.join(self.raw_folder, "train-labels-idx1-ubyte")
    #         ),
    #     )
    #     test_set = (
    #         read_image_file(
    #             os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")
    #         ),
    #         read_label_file(
    #             os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte")
    #         ),
    #     )
    #     with open(
    #         os.path.join(self.processed_folder, self.training_file), "wb"
    #     ) as f:
    #         torch.save(training_set, f)
    #     with open(
    #         os.path.join(self.processed_folder, self.test_file), "wb"
    #     ) as f:
    #         torch.save(test_set, f)
    #
    #     print("Done!")

    def get_file_paths(self):
        # TODO! split train, valid, test sets
        data_path = Path(os.path.join(self.data_path, self.dataset_type))
        for root, dirs, files in walk(data_path, topdown=False):
            for name in files:
                filename, file_ext = splitext(basename(name))
                file_ext = file_ext.lower()
                if file_ext == ".png" or file_ext == ".jpg":
                    self.image_paths.append(os.path.join(root, name))
                elif file_ext == ".txt":
                    self.image_metadata_txt_paths.append(path.join(root, name))
                elif file_ext == ".xml":
                    self.image_metadata_xml_paths.append(path.join(root, name))
                else:
                    assert False, "!!! Not supported !!!"
        return self.image_paths, self.image_metadata_txt_paths, self.image_metadata_xml_paths

    def read_images(self, plates_and_positions=None, image_paths=None, index_maximum=None, crop_plate=True):
        if image_paths is None:
            image_paths = self.image_paths
        if plates_and_positions is None:
            plates_and_positions = self.get_plates_and_positions()
        if index_maximum is None:
            index_maximum = self.index_maximum
        for image_path, plate_and_position in zip(image_paths[:index_maximum], plates_and_positions[:index_maximum]):
            self.read_image(image_path, plate_and_position, crop_plate)
        return self.images  # np.stack(images)

    def read_image(self, image_path, plate_and_position, crop_plate=True):
        image = cv2.imread(image_path)
        if crop_plate:
            image = self._crop_plate_func(image, plate_and_position)
        self.images.append(image)

    def write_crop_plate_func(self, image, position_plate):
        raise NotImplementedError

    def _crop_plate_func(self, image, position_plate):
        # plate = position_plate[0]
        position = position_plate[1]
        bounding_box = self.position_to_bounding_box(position)
        image = image[bounding_box[0, 0]: bounding_box[1, 0], bounding_box[0, 1]: bounding_box[1, 1]]
        return image

    def position_to_bounding_box(self, position):
        coords = [int(coord) for coord in position.split()]
        upper_left = [coords[1], coords[0]]
        lower_right = [coords[1] + coords[3], coords[0] + coords[2]]
        bounding_box = np.array([upper_left, lower_right])
        return bounding_box

    def read_metadata(self, image_metadata_paths=None):
        image_metadatas = []
        if image_metadata_paths is None:
            image_metadata_paths = self.image_metadata_txt_paths
        for image_metadata_path in image_metadata_paths:
            with open(image_metadata_path, "r") as image_metadata_file:
                image_metadata = image_metadata_file.readlines()
            image_metadatas.append(image_metadata)
        return image_metadatas

    def parse_metadata_txt(self, index_maximum=None, image_metadatas=None):
        if image_metadatas is None:
            image_metadatas = self.read_metadata()
        if index_maximum is None:
            index_maximum = self.index_maximum
        all_info = {}
        all_info_list = []
        for image_metadata in image_metadatas[:index_maximum]:
            # headlines
            camera = image_metadata[0].split(':')[-1][1:-1]
            position_vehicle = image_metadata[1].split(':')[-1][1:-2]
            # vehicle
            type = image_metadata[2].split(':')[-1][1:-1]
            make = image_metadata[3].split(':')[-1][1:-1]
            model = image_metadata[4].split(':')[-1][1:-1]
            year = image_metadata[5].split(':')[-1][1:-1]
            # plate
            plate = image_metadata[6].split(':')[-1][1:-1]
            position_plate = image_metadata[7].split(':')[-1][1:-1]
            # characters
            char_bbs = [meta.split(':')[-1][1:-1] for meta in image_metadata[8:]]

            all_info = {
                "camera": camera,
                "position_vehicle": position_vehicle,
                "type": type,
                "make": make,
                "model": model,
                "year": year,
                "plate": plate,
                "position_plate": position_plate,
                "char_bbs": char_bbs,
            }
            all_info_list.append(all_info)
        self.parsed_metadatas = all_info_list
        return all_info_list

    def get_plates_and_positions(self):
        plates_and_position = []
        for image_metadata in self.parsed_metadatas:
            # plate
            plate = image_metadata["plate"]
            position_plate = image_metadata["position_plate"]
            plates_and_position.append([plate, position_plate])
        return plates_and_position


if __name__ == "__main__":
    data_path = Path(r"D:\PycharmProjects\ocr_toolkit\UFPR-ALPR dataset")
    index_maximum = 5
    parser = UFPR_ALPR_dataset(data_path=data_path, dataset_type="train", crop_plate=True, index_maximum=index_maximum)  # , index_maximum=index_maximum
    # print(parser[0])
    plates_and_positions = parser.get_plates_and_positions()
    print(f"plates_and_positions: {plates_and_positions}%")

    # plate_and_position = parser[:3]
    plate_and_position = parser[0]
    print(f"plate_and_position: {plate_and_position}")
    

