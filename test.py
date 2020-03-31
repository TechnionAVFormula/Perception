import os
import ast
import cv2
import numpy as np
from PIL import Image
from detect import get_BB_from_img
from perception_functions import predict_cone_color, predict_cone_depth,trasform_img_cones_to_xyz

# convert from bit representation to RGB + depth format
img_RGB = Image.open('simulation data/four_cones_raw.jpg').convert('RGB')
img_depth = Image.open('simulation data/four_cones_depth.png')
width, height = img_RGB.width, img_RGB.height
h_fov = 50  # [deg]
v_fov = 29.394957  # [deg]

# set NN parameters
weights_path = 'outputs/february-2020-experiments/yolo_baseline/9.weights'
model_cfg = 'model_cfg/yolo_baseline.cfg'

# Detect cones in input image via YOLO

# get image cones: BB_list=[[x,y,w,h,type,depth],[x,y,w,h,type,depth],....]
# x,y - left top bounding box position in image plain
# w, h - width and height of bounding box in pixels
# type - cone color: 'B' - blue, 'Y' - yellow, 'O' - orange
# depth - nominal depth value
BB_list = get_BB_from_img(img_RGB,weights_path,model_cfg,conf_thres = 0.8,nms_thres = 0.25,xy_loss = 2,wh_loss = 1.6,no_object_loss = 25,object_loss=0.1,vanilla_anchor = False)
# classify detected cone to types
for BB in BB_list:
    cone_color = predict_cone_color(img_RGB, BB)
    cone_depth = predict_cone_depth(img_depth, BB)
    # BB = [x,y,w,h,type,depth]
    BB.append(cone_color)
    BB.append(cone_depth)
# print BB results
print("Bounding box list in image plain:")
for i, BB in enumerate(BB_list):
    print(f"({i}) x = {BB[0]}, y = {BB[1]}, w = {BB[2]}, h = {BB[3]}, type = {BB[4]}, depth = {BB[5]}")
# transformation from image plain to cartesian coordinate system
# xyz_cones = [(X, Y, Z, type), (X, Y, Z, type), ....]
# X,Y,Z - in ENU coordinate system (X - right, Y-forward, Z-upward)
# type - cone color: 'B' - blue, 'Y' - yellow, 'O' - orange
img_cones = BB_list
xyz_cones = trasform_img_cones_to_xyz(img_cones, img_depth, h_fov, v_fov, width, height)

# print XYZ results
print("Cones X,Y,Z list in ENU coordinate system (X - right, Y - forward, Z - upward):")
for i, xyz_cone in enumerate(xyz_cones):
    print(f"({i}) X = {int(xyz_cone[0])}, Y = {int(xyz_cone[1])}, Z = {int(xyz_cone[2])}, type = {xyz_cone[3]}")

