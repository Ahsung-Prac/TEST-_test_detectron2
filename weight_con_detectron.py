import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import torch
import matplotlib.pyplot as plt
import math
from imageTools import imageTool
import sys
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import VisImage
from detectron2.data import MetadataCatalog


if len(sys.argv) < 2:
    print("usage:python weight_con_detectron.py [input_Image]")
    sys.exit(0)

im = cv2.imread(sys.argv[1])

if  im is None:
    print("file open fail")
    sys.exit(0)


cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# start path in configs [dir]
fileName = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
model = model_zoo.get_config_file(fileName)

cfg.merge_from_file(model)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.merge_from_list(['MODEL.DEVICE','cpu'])

# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the detectron2:// shorthand
cfg.MODEL.WEIGHTS = "detectron2://"+model_zoo.get_weight_suffix(fileName)
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# if use ponoptic model..
#panoptic_seg, segments_info = outputs['panoptic_seg']

boxes = outputs["instances"].pred_boxes.tensor
pred = outputs['instances'].pred_classes
masks = outputs["instances"].pred_masks
scores = outputs["instances"].scores

# Get weight of importance of echo instance, and Main instance index
idx, weightlist = imageTool.get_weight(outputs, im,False)

print("weightList:",weightlist)

# concatenate close instace from Main_instance
conlist=imageTool.getconInstances(boxes,idx,weightlist,6,0.5)

print("concatenation imglist:",conlist)
print("Main:",idx)
print("pred_class:",pred[conlist])

# combine img_box
Y_S,Y_D,X_S,X_D = imageTool.combinde_img_box(boxes[conlist])

# combinMask
comebineMask = imageTool.combine_img_mask(masks[conlist])

rmbgImg = imageTool.rmBg(im,comebineMask,0,Y_S,Y_D,X_S,X_D)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
vtmp = v.get_image()[:, :, ::-1]


# if use ponoptic model..
# v2 = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
# v2 = v2.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
# vtmp2 = v2.get_image()[:, :, ::-1]


mx1,my1,mx2,my2 = boxes[idx]  #Main Instace box pos

#출력
#cv2.imshow('panotic',vtmp2)

cv2.imshow('predict',vtmp)
cv2.imshow('mask',rmbgImg)
cv2.imshow('result',imageTool.fitsize(im,Y_S,Y_D,X_S,X_D))
cv2.imshow('main',imageTool.fitsize(im,my1,my2,mx1,mx2))

#instance 별로 하나씩 출력

# n=0
# for b in boxes:
#     cv2.imshow(str(n),imageTool.fitsize(im,b[1],b[3],b[0],b[2]))
#     n+=1

# 연결된 instance하나씩

# for i in conlist:
#     cv2.imshow(str(i),imageTool.fitsize(im,boxes[i][1],boxes[i][3],boxes[i][0],boxes[i][2]))


cv2.waitKey(0)
cv2.destroyAllWindows()