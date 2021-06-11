import time

import pandas as pd
import tensorflow as tf

from src.dataset.coco import load_dataset
from src.definitions import OUTPUT_DIR
from src.main import load_tf_hub_model

tf.get_logger().setLevel("ERROR")

if __name__ == "__main__":

    coco_dataset = load_dataset()

    model_to_inference_time = dict()

    for model_name in [
        "Faster R-CNN Inception ResNet V2 1024x1024",
        "CenterNet HourGlass104 1024x1024",
        "EfficientDet D4 1024x1024",
        "SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)",
        "Mask R-CNN Inception ResNet V2 1024x1024",
    ]:
        hub_model = load_tf_hub_model(model_name)

        num_of_images = 100
        img_counter = 0

        model_to_inference_time[model_name] = []

        for sample in coco_dataset:
            if img_counter == num_of_images:
                break

            start_time = time.time()
            result = hub_model(tf.expand_dims(sample["image"], axis=0))
            end_time = time.time()

            inference_time = end_time - start_time
            model_to_inference_time[model_name].append(inference_time)

            img_counter += 1

    pd.DataFrame(model_to_inference_time).to_csv(OUTPUT_DIR, 'inference_times_across_models.csv', index=False)
