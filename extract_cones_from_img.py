import cv2
import os

def extract_cones_from_img(target_path, BB_list, type):
    if type == 'color':
        output_folder_path = 'C:/Users/Administrator/PycharmProjects/MIT_YOLO/temp/color'
    elif type == 'depth':
        output_folder_path = 'C:/Users/Administrator/PycharmProjects/MIT_YOLO/temp/depth'
    else:
        print("Invalid type!")
        return

    filelist = [f for f in os.listdir(output_folder_path)]
    for f in filelist:
        os.remove(os.path.join(output_folder_path, f))

    # target_path = 'dataset/YOLO_Dataset/v4_0058.png'
    # BB_list = [[498, 208, 44, 30], [144, 192, 42, 28]]  # x, y, h, w

    img = cv2.imread(target_path)
    idx = 0
    for BB in BB_list:
        # print("i = ", idx)
        # print("BB = ", BB)
        [x, y, h, w] = BB
        crop_img = img[y:y + h, x:x + w]
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        file_name = '/BB_' + str(idx) + '.png'
        # print(output_folder_path + file_name)
        cv2.imwrite(output_folder_path + file_name, crop_img)
        idx += 1
