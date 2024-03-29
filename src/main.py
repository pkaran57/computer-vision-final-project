import json
import os
import time

import cv2
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import visualization_utils as viz_utils

from all_models import ALL_MODELS
from src.dataset.coco import get_category_index, load_dataset
from src.definitions import OUTPUT_DIR
from src.metricFunctions import overall

tf.get_logger().setLevel("ERROR")


def load_tf_hub_model(model_name):
    print("loading model {} ...".format(model_name))
    module_handle = ALL_MODELS[model_name]
    hub_model = hub.load(module_handle)
    print("model loaded!")

    return hub_model


def get_image_with_predictions(original_image, result, category_index):
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

    model_stats = dict()

    for model_name in [
        "Faster R-CNN Inception ResNet V2 1024x1024",
        "CenterNet HourGlass104 1024x1024",
        "EfficientDet D4 1024x1024",
        "SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)",
        "Mask R-CNN Inception ResNet V2 1024x1024",
    ]:
        model_stats[model_name] = list()

        hub_model = load_tf_hub_model(model_name)

        num_of_images = 30
        img_counter = 0

        for sample in coco_dataset:
            if img_counter == num_of_images:
                break

            original_image = sample["image"]
            original_image_name = "original-{}.png".format(img_counter)
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, original_image_name),
                original_image.numpy(),
            )

            model_input = tf.expand_dims(original_image, axis=0)

            start_time = time.time()
            result = hub_model(model_input)
            end_time = time.time()

            inference_time = end_time - start_time

            # only output 5 images with prediction boxes overplayed on top of original images
            if img_counter < 5:
                image_with_predictions = get_image_with_predictions(original_image, result, category_index)

                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, "{}-output-image-{}.png".format(model_name, img_counter)),
                    image_with_predictions,
                )

            try:
                precision, recall = overall(sample["objects"], result, (sample['image'].shape[0], sample['image'].shape[1]))

                print(f'precision = {precision}, recall = {recall}')

                model_stats[model_name].append({
                    'image_name': original_image_name,
                    'inference_time': inference_time,
                    'precision': precision,
                    'recall': recall
                })
            except Exception as e:
                print('Error: ', e)

            img_counter += 1

    print("Final Result : \n\n", json.dumps(model_stats))
    pd.DataFrame(model_stats).to_csv(os.path.join(OUTPUT_DIR, 'model_stats.csv'), index=False)
