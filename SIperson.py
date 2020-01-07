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

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

im = cv2.imread(sys.argv[1])

if  im is None:
    print("file open fail")
    sys.exit(0)

#cv2.imshow('image',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.merge_from_list(['MODEL.DEVICE','cpu'])
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the detectron2:// shorthand
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

tim = im.copy()
be = np.where(masks[idx]!=True)
tim[be] = 255
tim[be] = 255
tim[be] = 255
tim = tim[int(y_s):int(y_d),int(x_s):int(x_d)]

cv2.imshow('imagin',tim)
cv2.waitKey(0)
cv2.destroyAllWindows()
