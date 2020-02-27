import cv2
import numpy as np
import os

def predict_cone_color():
    BB_folder_path = 'C:/Users/Administrator/PycharmProjects/MIT_YOLO/temp/color'
    frame_list = [f for f in os.listdir(BB_folder_path)]
    cones_color = []
    for frame_name in frame_list:
        frame_path = BB_folder_path + '/' + frame_name
        frame = cv2.imread(frame_path)
        # cv2.imshow("Img", frame)
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
        if max_idx == 0:
            print(frame_name, "| Cone color is Yellow")
            cones_color.append('Y')
        elif max_idx == 1:
            print(frame_name, "| Cone color is Blue")
            cones_color.append('B')
        else:
            print(frame_name, "| Cone color is Orange")
            cones_color.append('O')

    return cones_color
        # cv2.waitKey()
