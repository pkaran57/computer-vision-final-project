def IoU(gtbbox, pbbox):
  #gtbbox = ground truth bbox, gotten from coco [left x, top y, width, height]
  #pbbox = predicted bbox, gotten from out program, altered to [left x, top y, width, height]
  #final format = [left x, right x, top y, bottom y]

  gtBBoxArea = gtbbox[2] * gtbbox[3]
  gtBBox = [gtbbox[0], gtbbox[0] + gtbbox[2], gtbbox[1], gtbbox[1] + gtbbox[3]]
  pBBoxArea = pbbox[2] * pbbox[3]
  predBBox = [pbbox[0], pbbox[0] + pbbox[2], pbbox[1], pbbox[1] + pbbox[3]]
  

  xIntLeft = max(gtBBox[0], predBBox[0])
  xIntRight = min(gtBBox[1], predBBox[1])
  yIntUp = max(gtBBox[2], predBBox[2])
  yIntLow = min(gtBBox[3], predBBox[3])

  intArea = max(0, (xIntRight - xIntLeft)) * max(0, (yIntLow - yIntUp))

  IoU = intArea / (gtBBoxArea + pBBoxArea - intArea)
  return IoU


#for an individual picture
#returns (precision, recall)
def overall(gt, results, imShape, conThresh): 
  #gt is the objects variable from the samples loop (main.py, ln 41)
  #results is the variable from (main.py, ln 44)
  #imShape is in y,x (height,width) format. If you get this from image_np.shape, it's in (1, height, width, 3), so just do (image_np.shape[1], image_np.shape[2]).
    #if you find the image shape somewhere else in x,y, switch it in the input or in the next line of code
  #.5 is the min conThresh.
    #SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152) works around .5 and basically barely works above .7 (on the pic I tested)
    #Faster R-CNN Inception ResNet V2 1024x1024 works great around .9 (on the pic I tested)
    #you might have to figure a good threshold for each individual model
  #print statements are for likely issues that I can't check since I can't get the database, if things work just delete them all

  #convert imShape to x,y instead of y,x
  im = [imShape[1], imShape[0]]

  #set up counts for TP, FP, and number of predictions over the confidence threshold
  TP = 0
  FP = 0

  #using threshold, get how many outputs we care about
  numOverThresh = sum(result['detection_scores'][0] > conThresh)

  #retrieve labels/gtbboxes for objects in the pic
  gtlabels = gt["label"] #This should be a list of all gt labels for all objects in the image, if this doesn't work, "label" needs to be "catagory_id"
  gtbbox = gt["bbox"]     #should be a (n,4) list of bboxes, format = [left x, topy, width (x), height (y)]

  print("this should be the number of gt objects in the pic")
  print(gtbbox.shape[0])

  #retrieve results data from the program
  pBbox = result["detection_boxes"][0][:numOverThresh].numpy() #should be in form [ymin, xmin, ymax, xmax]
  pclasses = result["detection_classes"][0][:numOverThresh].astype(int)
  
  #convert from model bbox output (pBbox) to useable output (pbox)
  pbox = np.zeros(numOverThresh * 4).reshape(numOverThresh, 4)
  for i in range(numOverThresh):
    pbox[i][0] = pBbox[1]*im[0] #x_min * image width
    pbox[i][1] = pBbox[0]*im[1] #y_min * image height
    pbox[i][2] = pBbox[3]*im[0] - pbox[i][0] #x_max * image width - x_min * image width = box width
    pbox[i][3] = pBbox[2]*im[1] - pbox[i][1] #y_max * image height - y_min * image height = box height
  #this should end with pbox[i] having format [xmin, ymin, width, height]

  #determine which pbboxes to compare to which gtbboxs 
  #for every gtbox
  for gt in range(gtbbox.shape[0]):
    #check the label of each pbox, if ==, 
    for p in range(numOverThresh):

      print("if these are coming out ALL false, switch the commented line and add label_id_offset to the function input")
      print(gtlabels[gt] == (result['detection_classes'][0][p]).astype(int))

      if (gtlabels[gt] == (result['detection_classes'][0][p]).astype(int)):
#      if (gtlabels[gt] == (result['detection_classes'][0][p] + label_id_offset).astype(int)): #label_id_offset is 0 in the main function and never gets changed, I don't think we need it
        #check IoU, if >.5
        if (IoU(gtbbox[gt], pbox[p]) > .5):   #using .5 as the threshold for what is an acceptable IoU,
          TP += 1
  # set the false positives to be equal to the number of pboxes that haven't been a TP for a gtbbox
  FP = numOverThresh - TP

  #calc precision and recall for the picture 
  precision = TP / (TP + FP)
  recall = TP / gtbbox.shape[0]   #recall = TP / (TP + FN), TP + FN = number of true objects

  return (precision, recall)

