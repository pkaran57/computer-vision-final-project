import os

import cv2
import tensorflow as tf
import tensorflow_hub as hub
from object_detection.utils import visualization_utils as viz_utils

from all_models import ALL_MODELS
from src.dataset.coco import get_category_index, load_dataset
from src.definitions import OUTPUT_DIR
from src.metricFunctions import overall

# New imports
import time
from src.metricFunctions_2 import *

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

    # To save metrics for every model
    metrics_info = {}

    for model_name in [
        "Faster R-CNN Inception ResNet V2 1024x1024",
        "CenterNet HourGlass104 1024x1024",
        "EfficientDet D4 1024x1024",
        "SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)",
        "Mask R-CNN Inception ResNet V2 1024x1024",
    ]:
        start_time = time.time()  # Added this line to track the total time taken for a model
        hub_model = load_tf_hub_model(model_name)

        num_of_images = 5
        img_counter = 0

        # Te save the TP, FP, and FN dicts
        image_results = {}

        for sample in coco_dataset:
            if img_counter == num_of_images:  # Changed the hard-coded value '5' to 'num_of_images'
                break

            original_image = sample["image"]

            # Added code (for more clarity)
            original_image_id = sample["image/id"]
            original_image_objects_areas = sample["objects"]["area"]
            original_image_objects_bboxes = sample["objects"]["bbox"]
            # =======================================================

            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "original-{}.png".format(img_counter)),
                original_image.numpy(),
            )

            result = hub_model(tf.expand_dims(original_image, axis=0))

            image_with_predictions = get_image_with_predictions(original_image, result, category_index)

            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "{}-output-image-{}.png".format(model_name, img_counter)),
                image_with_predictions,
            )

            # Added code (for more clarity)
            pred_image_detection_boxes = result["detection_boxes"]
            pred_image_detection_scores = result["detection_scores"]
            pred_image_detection_classes = result["detection_classes"]
            # ==========================================================

            # Previous code
            # precision, recall = overall(sample["objects"], result, (sample['image'].shape[0], sample['image'].shape[1]))
            # print(precision, recall)
            # =============================================================

            # Replaced code

            # Get single image results
            gt_boxes = original_image_objects_bboxes.numpy()
            pred_boxes = pred_image_detection_boxes.numpy()

            pos_and_neg_dict = get_single_image_results(gt_boxes=gt_boxes,
                                                        pred_boxes=np.squeeze(pred_boxes, axis=0),
                                                        iou_thr=0.5)

            # Save the TP, FP, and FN dict in image_results
            image_results[original_image_id.numpy().astype(int)] = pos_and_neg_dict

            # =============================================================

            img_counter += 1

        # New Code for calculating precision and recall after getting TP, FP, FN for individual images
        # Calculate precision and recall for the current selected model
        precision, recall = calc_precision_recall(image_results)

        # Save the precision and recall in "metrics_info" dict
        metrics_info[model_name] = {
            "precision": precision,
            "recall": recall,
        }
        print("Done..")
        print(metrics_info)
        end_time = time.time()
        print("Inference time..{}s".format(round(end_time - start_time, 3)))
        # ==========================================================================
