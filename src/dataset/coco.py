import json
import os

import tensorflow_datasets as tfds

from src.definitions import DATA_DIR


def get_category_index():
    with open(os.path.join(DATA_DIR, "label-map.json")) as json_file:
        category_index = json.load(json_file)
        return {int(k): v for k, v in category_index.items()}


def load_dataset():
    print("loading dataset ...")
    # https://www.tensorflow.org/datasets/catalog/coco_captions
    coco_dataset = tfds.load("coco_captions", split="train[:5%]")
    print("dataset loaded! : {}".format(coco_dataset))

    return coco_dataset
