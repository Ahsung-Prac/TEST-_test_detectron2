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
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def binarysearch(mask,x_s,x_d,flag):
    maskArry = torch.where(mask == True)[0]
    if maskArry.size() == torch.Size([0]):
        return -1
    if flag :
        src = min(maskArry)+1
    else : src = max(maskArry)-1
    return src


def expand(masks,size,y_s,y_d,x_s,x_d):
    mask = masks.clone()
    masktemp = masks.clone()
    y_s,y_d,x_s,x_d = int(y_s),int(y_d),int(x_s),int(x_d)
    for i in range(y_s,y_d):
        leftx = binarysearch(masktemp[i],x_s,x_d,True)
        rightx = binarysearch(masktemp[i],x_s,x_d,False)
        if leftx == -1 :
            continue
        #print(leftx,rightx)
        for j in range(size):
            mask[i][leftx-j] = True    #왼쪽으로 쭉          
            mask[i][rightx+j] = True

    masktemp = masks.clone()
    for i in range(x_s,x_d):
        upy = binarysearch(masktemp.T[i],y_s,y_d,True)
        downy = binarysearch(masktemp.T[i],y_s,y_d,False)
        if upy == -1 :
            continue
        for j in range(size):
            mask[upy-j][i] = True    #위로 쭉
            mask[downy+j][i] = True
    return mask



#if__name__

if len(sys.argv) < 2:
    print("usage:python Allperson.py [input_Image]")
    sys.exit(0)

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

#rmBgperson
tim = im.copy()
be = torch.where(masks[idx]!=True)
tim[be] = 224

# detectron Visualizer
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
vtmp = v.get_image()[:,:,::-1]

#boxPerson
box = im[int(y_s):int(y_d),int(x_s):int(x_d)]

#expand rmBGperson
etim = torch.from_numpy(im.copy())
emask = expand(masks[idx],20,y_s,y_d,x_s,x_d)
ebe = torch.where(emask!=True)
etim[ebe] = 224
etim = etim.numpy()

#Blur
gauss = cv2.GaussianBlur(im,(5,5),1e+10)
median = cv2.medianBlur(im,5)
tr = torch.where(masks[idx] == True)
gauss[tr] = im[tr]
median[tr] = im[tr]


cv2.imshow('median',median)
cv2.imshow('Gauss',gauss)
cv2.imshow('expand rmBg',etim)
cv2.imshow('box',box)
cv2.imshow('predic',vtmp)
cv2.imshow('rmBgperson',tim)
cv2.waitKey(0)
cv2.destroyAllWindows()
