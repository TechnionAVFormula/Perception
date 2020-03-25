import os
import ast
import cv2
import numpy as np
from PIL import Image
from detect import get_BB_from_img
from perception_functions import predict_cone_color, predict_cone_depth

# convert from bit representation to RGB + depth format
img_RGB = Image.open('0000000066.png').convert('RGB')
img_depth = img_RGB
# set NN parameters
weights_path = 'outputs/february-2020-experiments/yolo_baseline/9.weights'
model_cfg = 'model_cfg/yolo_baseline.cfg'

# Detect cones in input image via YOLO
BB_list = get_BB_from_img(img_RGB,weights_path,model_cfg,conf_thres = 0.8,nms_thres = 0.25,xy_loss = 2,wh_loss = 1.6,no_object_loss = 25,object_loss=0.1,vanilla_anchor = False)

# classify detected cone to types
for BB in BB_list:
    cone_color = predict_cone_color(img_RGB, BB)
    cone_depth = predict_cone_depth(img_depth, BB)
    # BB = [x,y,w,h,type,depth]
    BB.append(cone_color)
    BB.append(cone_depth)
# BB_list = [BB_1,BB_2,...,BB_N]
print(BB_list)