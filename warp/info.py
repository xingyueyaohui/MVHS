"""
This is the file for extracting information from json, video and images
"""

import json
import numpy as np
import cv2
import os

"""
Extract pose information of a specific person 
Argument:
    json_path: json file path
    img_dir: path of image directory
    index: person number
Return value:
    info in form of {'img', 'joints', 'id'}
Possible modifications:
    joints number be 17
    Order of joints in form of alpha-pose
"""
def make_person_pose_info(json_path, img_dir, index):
    file = open(json_path)
    obj = json.load(file)
    person = obj[index]
    img = cv2.imread(img_dir + person['img_id'])

    joints_num = 17
    joints = np.zeros((joints_num, 2))
    for i in range(joints_num):
        joints[i, 1] = person['keypoints'][3 * i + 1]  # Direction of height
        joints[i, 0] = person['keypoints'][3 * i]  # Direction of width

    info = {'img': img, 'joints': joints, 'id': person['image_id']}
    return info

