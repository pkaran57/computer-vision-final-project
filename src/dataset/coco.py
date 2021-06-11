import json
import os

import tensorflow_datasets as tfds

from src.definitions import DATA_DIR


def get_category_index():
    with open(os.path.join(DATA_DIR, "label-map.json")) as json_file:
        category_index = json.load(json_file)
        return {int(k): v for k, v in category_index.items()}


def get_label_to_id_map_coco_paper():
    with open(os.path.join(DATA_DIR, "coco-labels-paper.txt")) as file:
        labels = file.readlines()
        return {labels[id].strip(): id for id in range(len(labels))}


def get_id_to_label_map_coco_dataset():
    with open(os.path.join(DATA_DIR, "coco-labels-2014_2017.txt")) as file:
        labels = file.readlines()
        return {id: labels[id].strip() for id in range(len(labels))}


def load_dataset():
    print("loading dataset ...")
    # https://www.tensorflow.org/datasets/catalog/coco_captions
    coco_dataset = tfds.load("coco_captions", split="train[:5%]")
    print("dataset loaded! : {}".format(coco_dataset))

    return coco_dataset
