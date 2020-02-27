import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

def predict_cone_depth():
    BB_folder_path = 'C:/Users/Administrator/PycharmProjects/MIT_YOLO/temp/depth'
    frame_list = [f for f in os.listdir(BB_folder_path)]
    cones_depth = []
    for frame_name in frame_list:
        frame_path = BB_folder_path + '/' + frame_name
        frame = cv2.imread(frame_path)
        # cv2.imshow("Img", frame)
        # histg = cv2.calcHist([frame], [0], None, [256], [0, 256])
        # plt.plot(histg)
        # plt.show()
        # cv2.waitKey()
        array = np.array(frame)
        max_pixel_value = array.max()
        print(frame_name, "| Cone depth is ", max_pixel_value)
        cones_depth.append(max_pixel_value)

    return cones_depth
