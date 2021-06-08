import json
import os

import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from object_detection.utils import visualization_utils as viz_utils

tf.get_logger().setLevel("ERROR")

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "data", "label-map.json")) as json_file:
    category_index = json.load(json_file)
    category_index = {int(k): v for k, v in category_index.items()}

print("loading dataset ...")
# https://www.tensorflow.org/datasets/catalog/coco_captions
coco_dataset = tfds.load("coco_captions", split="train[:5%]")
print("dataset loaded! : {}".format(coco_dataset))

print("loading model...")
module_handle = (
    "https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1"
)
hub_model = hub.load(module_handle)
print("model loaded!")

for sample in coco_dataset:
    print(
        "Following are keys associated with each sample from coco dataset: {}",
        list(sample.keys()),
    )

    captions = sample["captions"]
    image = sample["image"]
    image_file_name = sample["image/filename"]
    image_id = sample["image/id"]
    objects = sample["objects"]

    cv2.imwrite(os.path.join(dir_path, "output", "original.png"), image.numpy())

    # plt.plot(image.numpy())
    # plt.savefig(os.path.join(dir_path, 'output', 'original.png'))

    result = hub_model(tf.expand_dims(image, axis=0))

    print("test")
    #
    label_id_offset = 0
    image_np_with_detections = image.numpy().copy()

    # Use keypoints if available in detections
    # keypoints, keypoint_scores = None, None
    # if 'detection_keypoints' in result:
    #     keypoints = result['detection_keypoints'][0]
    #     keypoint_scores = result['detection_keypoint_scores'][0]

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        result["detection_boxes"][0].numpy(),
        result["detection_classes"][0].numpy().astype(int),
        result["detection_scores"][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        agnostic_mode=False,
    )
    # keypoints=keypoints)
    # keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

    # plt.figure(figsize=(24, 32))
    # plt.plot(image_np_with_detections)
    # plt.savefig(os.path.join(dir_path, 'output', 'output-image.png'))
    # plt.show()

    cv2.imwrite(
        os.path.join(dir_path, "output", "output-image.png"), image_np_with_detections
    )

    break
