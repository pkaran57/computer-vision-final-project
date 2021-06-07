import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from matplotlib import pyplot as plt

tf.get_logger().setLevel("ERROR")

print("loading dataset ...")
coco_dataset = tfds.load("coco_captions", split="train[:5%]")
print("dataset loaded! : {}".format(coco_dataset))

print("loading model...")
module_handle = 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1'
hub_model = hub.load(module_handle)
print('model loaded!')

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

    plt.imshow(image.numpy())
    plt.show()

    result = hub_model(tf.expand_dims(image, axis=0))

    print('test')
    #
    # label_id_offset = 0
    # image_np_with_detections = image.numpy().copy()
    #
    # # Use keypoints if available in detections
    # keypoints, keypoint_scores = None, None
    # if 'detection_keypoints' in result:
    #     keypoints = result['detection_keypoints'][0]
    #     keypoint_scores = result['detection_keypoint_scores'][0]
    #
    #     viz_utils.visualize_boxes_and_labels_on_image_array(
    #     image_np_with_detections[0],
    #     result['detection_boxes'][0],
    #     (result['detection_classes'][0] + label_id_offset).astype(int),
    #     result['detection_scores'][0],
    #     category_index,
    #     use_normalized_coordinates=True,
    #     max_boxes_to_draw=200,
    #     min_score_thresh=.30,
    #     agnostic_mode=False,
    #     keypoints=keypoints,
    #     keypoint_scores=keypoint_scores,
    #     keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)
    #
    #     plt.figure(figsize=(24,32))
    #     plt.imshow(image_np_with_detections[0])
    #     plt.show()

    break
