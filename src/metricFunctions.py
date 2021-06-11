import numpy as np

from src.dataset.coco import get_id_to_label_map_coco_dataset, get_label_to_id_map_coco_paper

#this works if both gtbbox and pbbox are in [xmin,ymin,xmax,ymax] normalized form, and imShape is (x,y)
def IoU(gtbbox, pbbox, imShape):
  #final format = [left x, top y, right x, bottom y]
  #pbbox should be in pixels
  #gtbbox should be normalized

  #gtnormalized -> pixels
  gtBBox = [gtbbox[0]*imShape[0], gtbbox[1]*imShape[1], gtbbox[2]*imShape[0], gtbbox[3]*imShape[1]]
  predBBox = [pbbox[0], pbbox[1], pbbox[2]+pbbox[0], pbbox[3]+pbbox[1]]

  #gt area = width*height
  gtBBoxArea = (gtBBox[2] - gtBBox[0]) * (gtBBox[3] - gtBBox[1])
  pBBoxArea = (predBBox[2] - predBBox[0]) * (predBBox[3] - predBBox[1])

  xIntLeft = max(gtBBox[0], predBBox[0])
  yIntUp = max(gtBBox[1], predBBox[1])
  xIntRight = min(gtBBox[2], predBBox[2])
  yIntLow = min(gtBBox[3], predBBox[3])

  intArea = max(0, (xIntRight - xIntLeft)) * max(0, (yIntLow - yIntUp))

  IoU = intArea / (gtBBoxArea + pBBoxArea - intArea)
  return IoU
   


# for an individual picture
# returns (precision, recall)
def overall(gt, results, imShape, conThresh=0.5):
    # gt is the objects variable from the samples loop (main.py, ln 41)
    # results is the variable from (main.py, ln 44)
    # imShape is in y,x (height,width) format. If you get this from image_np.shape, it's in (1, height, width, 3), so just do (image_np.shape[1], image_np.shape[2]).
    # if you find the image shape somewhere else in x,y, switch it in the input or in the next line of code
    # .5 is the min conThresh.
    # SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152) works around .5 and basically barely works above .7 (on the pic I tested)
    # Faster R-CNN Inception ResNet V2 1024x1024 works great around .9 (on the pic I tested)
    # you might have to figure a good threshold for each individual model
    # print statements are for likely issues that I can't check since I can't get the database, if things work just delete them all

    # convert imShape to x,y instead of y,x
    im = [imShape[1], imShape[0]]

    # set up counts for TP, FP, and number of predictions over the confidence threshold
    TP = 0
    FP = 0

    # using threshold, get how many outputs we care about
    numOverThresh = sum(results["detection_scores"][0].numpy() > conThresh)

    # retrieve labels/gtbboxes for objects in the pic
    gtlabels = gt[
        "label"
    ]  # This should be a list of all gt labels for all objects in the image, if this doesn't work, "label" needs to be "catagory_id"
    gtbbox = gt[
        "bbox"
    ]  # should be a (n,4) list of bboxes, format = [left x, topy, width (x), height (y)]

    print("this should be the number of gt objects in the pic")
    print(gtbbox.shape[0])

    # retrieve results data from the program
    pBbox = results["detection_boxes"][0].numpy()[
            :numOverThresh
            ]  # should be in form [ymin, xmin, ymax, xmax]
    pclasses = results["detection_classes"][0].numpy()[:numOverThresh].astype(int)

    # convert from model bbox output (pBbox) to useable output (pbox)
    pbox = np.zeros(numOverThresh * 4).reshape(numOverThresh, 4)
    for i in range(numOverThresh):
        pbox[i][0] = pBbox[i][1] * im[0]  # x_min * image width
        pbox[i][1] = pBbox[i][0] * im[1]  # y_min * image height
        pbox[i][2] = (
                pBbox[i][3] * im[0] - pbox[i][0]
        )  # x_max * image width - x_min * image width = box width
        pbox[i][3] = (
                pBbox[i][2] * im[1] - pbox[i][1]
        )  # y_max * image height - y_min * image height = box height
    # this should end with pbox[i] having format [xmin, ymin, width, height] pixel

    id_to_label_map_coco_dataset = get_id_to_label_map_coco_dataset()
    label_to_id_map_coco_paper = get_label_to_id_map_coco_paper()

    # determine which pbboxes to compare to which gtbboxs
    # for every gtbox
    for gt in range(gtbbox.shape[0]):
        # check the label of each pbox, if ==,
        for p in range(numOverThresh):

            is_prediction_correct = label_to_id_map_coco_paper[id_to_label_map_coco_dataset[gtlabels[gt].numpy()]] + 1 == (results["detection_classes"][0].numpy()[p]).astype(
                int)

            print(
                "if these are coming out ALL false, switch the commented line and add label_id_offset to the function input"
            )

            print(is_prediction_correct)

            if is_prediction_correct:
                #      if (gtlabels[gt] == (result['detection_classes'][0][p] + label_id_offset).astype(int)): #label_id_offset is 0 in the main function and never gets changed, I don't think we need it
                # check IoU, if >.5
                if (
                        IoU(gtbbox[gt], pbox[p], im) > 0.5
                ):  # using .5 as the threshold for what is an acceptable IoU,
                    TP += 1
    # set the false positives to be equal to the number of pboxes that haven't been a TP for a gtbbox
    FP = numOverThresh - TP

    # calc precision and recall for the picture
    precision = TP / (TP + FP)
    recall = (
            TP / gtbbox.shape[0]
    )  # recall = TP / (TP + FN), TP + FN = number of true objects

    return (precision, recall)
