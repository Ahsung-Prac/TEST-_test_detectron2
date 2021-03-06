import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
from imageTools import imageTool
import sys
import torch
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


#test
if len(sys.argv) < 2:
    print("usage:python weight_con_detectron.py [input_Image]")
    sys.exit(0)

im = cv2.imread(sys.argv[1])

if  im is None:
    print("file open fail")
    sys.exit(0)

print('image shape :',im.shape,"\n")

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

if weightlist.size() == torch.Size([0]):
    print("No instance")
    sys.exit(0)

print("weightList:",weightlist)

# concatenate close instace from Main_instance
conlist=imageTool.getconInstances(boxes,idx,weightlist,6)

print("concatenation imglist:",conlist)
print("Main_index:",idx)
print("pred_class_list:",pred[conlist])

# combine img_box
Y_S,Y_D,X_S,X_D = imageTool.combinde_img_box(boxes[conlist])

# combinMask
comebineMask = imageTool.combine_img_mask(masks[conlist])

# mask 외에 size 매개변수만큼만 남겨둔후 remove background
rmbgImg = imageTool.rmBg(im,comebineMask,0,Y_S,Y_D,X_S,X_D)

#Main Instace box pos
mx1,my1,mx2,my2 = boxes[idx]

#사람중 가장 y축이 위에 있는 instance의 y축
peoplelist = torch.where(pred[conlist]==0)[0]
if peoplelist.size()==  torch.Size([0]):
    miny = Y_S
else:
    miny = torch.min(boxes[conlist[peoplelist],1],axis = 0)
    miny = miny.values



# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
vtmp = v.get_image()[:, :, ::-1]

vtmp = imageTool.limitsize(vtmp)
rmbgImg = imageTool.limitsize(rmbgImg)
result = imageTool.limitsize(imageTool.fitsize(im, Y_S, Y_D, X_S, X_D))
main = imageTool.limitsize(imageTool.fitsize(im, my1, my2, mx1, mx2))
rate16_9 =  imageTool.limitsize(imageTool.rate16_9(im,miny,Y_D,X_S,X_D))

cv2.imshow('16_9',rate16_9)
cv2.imshow('predict',vtmp)
cv2.imshow('mask',rmbgImg)
cv2.imshow('result',result)
cv2.imshow('main',main)


## instances 모두 다찍기!

# n=0
# for b in boxes:
#     cv2.imshow(str(n),imageTool.fitsize(im,b[1],b[3],b[0],b[2]))
#     n+=1
#
# #연결된 instance하나씩
#
# for i in conlist:
#     cv2.imshow(str(i),imageTool.fitsize(im,boxes[i][1],boxes[i][3],boxes[i][0],boxes[i][2]))


cv2.waitKey(0)
cv2.destroyAllWindows()
