# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import sys
import torch
import expandMask
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

#if__name__="main":

if len(sys.argv) < 3:
    print("usage:python expand_rmBgperson.py [input_Image] [expand_size]") 
    sys.exit(0)

im = cv2.imread(sys.argv[1])

if  im is None:
    print("file open fail")
    sys.exit(0)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.merge_from_list(['MODEL.DEVICE','cpu'])
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the detectron2:// shorthand

#config file info weight
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im) ; size = outputs["instances"].scores.shape[0];

poslist = []
boxes = outputs["instances"].pred_boxes.tensor
pred = outputs['instances'].pred_classes
masks = outputs["instances"].pred_masks

for i in range(size):
    if pred[i] == 0:
        wide = abs(boxes[i][2] - boxes[i][0])*abs(boxes[i][3]-boxes[i][1])
        poslist.append(int(wide))
    else:
        poslist.append(0)

print(poslist)

if poslist == [] :
    print("No person")
    sys.exit(0)

idx = poslist.index(max(poslist))

if poslist[idx] == 0:
    print("Not person")

x_s,y_s,x_d,y_d = boxes[idx]

expand_size = int(sys.argv[2])
etim = torch.from_numpy(im.copy())
emask = expandMask.expand(masks[idx],expand_size,y_s,y_d,x_s,x_d)
ebe = torch.where(emask!=True)
etim[ebe] = 224
etim = etim.numpy()
etim.shape


cv2.imshow('imagin',etim)
cv2.waitKey(0)
cv2.destroyAllWindows()
