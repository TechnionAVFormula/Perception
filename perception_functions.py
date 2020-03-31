import os
import ast
import cv2
import numpy as np
from PIL import Image
from detect import get_BB_from_img

def get_cones_from_camera(width, height, pixels):
    # convert from bit representation to RGB + depth format
    img_RGB, img_depth = convert_img_bits_to_RGBD(width, height, pixels)
    # set NN parameters
    weights_path = 'outputs/february-2020-experiments/yolo_baseline/9.weights'
    model_cfg = 'model_cfg/yolo_baseline.cfg'

    # Detect cones in input image via YOLO
    BB_list = get_BB_from_img(img_RGB,weights_path,model_cfg,conf_thres = 0.8,nms_thres = 0.25,xy_loss = 2,wh_loss = 1.6,no_object_loss = 25,vanilla_anchor = False)

    # classify detected cone to types
    for BB in BB_list:
        cone_color = predict_cone_color(img_RGB,BB)
        cone_depth = predict_cone_depth(img_depth,BB)
        # BB = [x,y,w,h,type,depth]
        BB.append(cone_color,cone_depth)
    # BB_list = [BB_1,BB_2,...,BB_N]
    return BB_list

def convert_img_bits_to_RGBD(width, height, pixels):
    # convert bit format to RGB
    img_RGB = Image.frombytes("RGB", (width, height), pixels, 'raw', 'RGBX', 0,-1)
    # convert bit format to depth channel
    CHANNEL_COUNT = 4
    frames = np.array(pixels)
    deinterleaved = [frames[idx::CHANNEL_COUNT] for idx in range(CHANNEL_COUNT)]
    img_depth = deinterleaved[3]

    return img_RGB, img_depth

def cut_cones_from_img(target_img, BB):
    [x, y, h, w] = BB
    crop_img = target_img.crop((x,y,x+w,y+h))
    return crop_img

def predict_cone_color(target_img, BB):

    frame = cut_cones_from_img(target_img, BB)
    frame = np.array(frame) # convert from PIL to cv
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # General mask
    low_general = np.array([0, 50, 80])
    high_general = np.array([179, 255, 255])
    general_mask = cv2.inRange(hsv_frame, low_general, high_general)
    general = cv2.bitwise_and(frame, frame, mask=general_mask)
    # cv2.imshow("General", general)

    # Yellow color
    low_yellow = np.array([20, 190, 20])
    high_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(general, general, mask=yellow_mask)
    # cv2.imshow("Yellow", yellow)

    # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(general, general, mask=blue_mask)
    # cv2.imshow("Blue", blue)

    # Orange color
    low_orange = np.array([5, 50, 50])
    high_orange = np.array([10, 295, 295])
    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
    orange = cv2.bitwise_and(general, general, mask=orange_mask)
    # cv2.imshow("Orange", orange)

    final_frame = cv2.hconcat((frame, yellow, blue, orange))
    cv2.imshow("final_frame", final_frame)

    n_white_pix_yellow = np.sum(yellow_mask == 255)
    n_white_pix_blue = np.sum(blue_mask == 255)
    n_white_pix_orange = np.sum(orange_mask == 255)
    n_white_pix = [n_white_pix_yellow, n_white_pix_blue, n_white_pix_orange]
    # print('Number of white pixels: Yellow_mask = ', n_white_pix_yellow, " | Blue_mask = ", n_white_pix_blue, " | Orange_mask = ", n_white_pix_orange)
    max_value = max(n_white_pix)
    max_idx = n_white_pix.index(max_value)

    if max_idx == 0: # cone is yellow
        return 'Y'
    elif max_idx == 1: # cone is blue
        return 'B'
    else: # cone is orange
        return 'O'

def predict_cone_depth(img_depth, BB):
    frame = img_depth
    array = np.array(frame)
    max_pixel_value = array.max()
    return max_pixel_value