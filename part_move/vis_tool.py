'''
Visualization tools,
Including:
    key points
    skeleton
'''

import json
import cv2
import numpy as np

def vis_joints(info, src_dir, dst_dir):
    """
    Visualize key points
    :param info: information for one person
    :param src_dir: image from
    :param dst_dir: image to
    :return:
    """

    img_name = info['id']
    img = cv2.imread(src_dir + img_name)
    joints = np.array(info['joints'])

    [n_joints, _] = joints.shape

    for j in range(17):
        x = int(joints[j][0])
        y = int(joints[j][1])
        cv2.circle(img, (x, y), 2, (255, 255, 255))

    cv2.imwrite(dst_dir + 'k' + img_name, img)