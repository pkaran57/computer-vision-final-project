import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

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

    result = hub_model(image)

    print(result)
    break
