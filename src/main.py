import os

import cv2
import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import visualization_utils as viz_utils

from all_models import ALL_MODELS
from src.dataset.coco import get_category_index, load_dataset
from src.definitions import OUTPUT_DIR

tf.get_logger().setLevel("ERROR")


def load_tf_hub_model(model_name):
    print("loading model {} ...".format(model_name))
    module_handle = ALL_MODELS[model_name]
    hub_model = hub.load(module_handle)
    print("model loaded!")

    return hub_model


def get_image_with_predictions(original_image, result):
    image_with_predictions = original_image.numpy().copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_predictions,
        result["detection_boxes"][0].numpy(),
        result["detection_classes"][0].numpy().astype(int),
        result["detection_scores"][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        agnostic_mode=False,
    )

    return image_with_predictions


if __name__ == "__main__":

    category_index = get_category_index()
    coco_dataset = load_dataset()

    for model_name in ["Faster R-CNN Inception ResNet V2 1024x1024"]:
        hub_model = load_tf_hub_model(model_name)

        for sample in coco_dataset:
            original_image = sample["image"]
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "{}-original.png".format(model_name)),
                original_image.numpy(),
            )

            result = hub_model(tf.expand_dims(original_image, axis=0))

            image_with_predictions = get_image_with_predictions(original_image, result)

            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "{}-output-image.png".format(model_name)),
                image_with_predictions,
            )

            break
