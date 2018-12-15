'''
Convert json format pose file
Including:
    alpha-pose format
'''
import numpy as np
import json

key_point_num = {'alpha':17}


def sort_json(json_path, dst_path, mode = 'alpha'):
    """
    Convert json into more handy format
    Delete repeated detection
    Sort according to image names
    :param json_path: src path
    :param dst_path: dst path
    :param mode: default alpha-pose
    """
    f = open(json_path, 'r')
    dic = json.load(f)
    f.close()
    res = []
    num = len(dic)
    joints_num = key_point_num['alpha']
    prev_img = None
    if mode == 'alpha':
        for i in range(num):
            joints = np.zeros((joints_num, 2))
            for j in range(joints_num):
                # x -- 0, y -- 1
                joints[j, 0] = dic[i]['keypoints'][3 * j + 0]
                joints[j, 1] = dic[i]['keypoints'][3 * j + 1]

            if dic[i]['image_id'] != prev_img:
                r = {'id': dic[i]['image_id'], 'joints': joints.tolist()}
                res.append(r)
                prev_img = dic[i]['image_id']
        res = sorted(res, key=lambda x: x['id'])

        f = open(dst_path, 'w')
        json.dump(dic, f)
        f.close()