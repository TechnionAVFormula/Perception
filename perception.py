from extract_cones_from_img import *
from predict_cone_color import *
from predict_cone_depth import *
import os
import ast

target_path = 'dataset/YOLO_Dataset/v4_0058.png'  # RGB input image path
target_depth_path = 'dataset/YOLO_Dataset/v4_0058_depth.png'  # input depth image path

# Detect cones in input image via YOLO
cmd = 'python detect.py --model_cfg=model_cfg/yolo_baseline.cfg --target_path=' + target_path + ' --weights_path=outputs/february-2020-experiments/yolo_baseline/9.weights'
os.system(cmd)

# define an empty BB list
BB_list = []

# open file (YOLO results) and read the content in a list
with open('temp/BB.txt', 'r') as filehandle:
    for line in filehandle:
        currentBB = line[:-1]
        currentBB = ast.literal_eval(currentBB)
        BB_list.append(currentBB)
# print("BB_list : ", BB_list)

# extract cones from RGB img. saved at 'C:/Users/Administrator/PycharmProjects/MIT_YOLO/temp/color'
extract_cones_from_img(target_path, BB_list, 'color')
# predict cones colors
cones_color = predict_cone_color()

# extract cones from depth img. saved at 'C:/Users/Administrator/PycharmProjects/MIT_YOLO/temp/depth'
extract_cones_from_img(target_depth_path, BB_list, 'depth')
# # predict cones depths
cones_depth = predict_cone_depth()

output = []
idx = 0
for BB in BB_list:
    x, y, h, w = BB
    x_c = round(x + w/2)  # center of BB
    y_c = round(y + h/2)  # center of BB
    z = cones_depth[idx]
    cone_type = cones_color[idx]
    temp_list = [x_c, y_c, z, cone_type]
    output.append(temp_list)
    idx += 1

print(output)