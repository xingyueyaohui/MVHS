"""
Do the pre work before training
to speed up training process
Including:
    reading the json and make up a mapping dictionary
    read in the images
    read in the pose map and target mask
    do transformations and write into file
    read the transformations
"""
import numpy as np
import pickle
import cv2
import json
import lib.move_util as move
import lib.data_generator as dg
from lib.dataset import Person, Dataset


def check_json(data_json_path, match_json_path, dst_path):
    f = open(match_json_path, 'r')
    match = json.load(f)
    f.close()
    f = open(data_json_path, 'r')
    data = json.load(f)
    f.close()

    img_num = len(data)
    dic = []
    i = 0
    for sample in data:
        if len(match[sample['id']]) >= 2:
            dic.append(sample)
    f = open(dst_path, 'w')
    json.dump(dic, f)

def construct_mapping(json_path):
    f = open(json_path, 'r')
    dic = json.load(f)
    img_num = len(dic)
    mapping = {}
    for i, obj in enumerate(dic):
        mapping[obj['id']] = i
    return mapping


def read_color(json_path, width, height, src_dir, session_num = None):
    """
    Read in colorful images like src
    :param json_path: pose json
    :param src_dir: image from
    :return: images in (num, 3, h, w)
    """
    f = open(json_path, 'r')
    dic = json.load(f)
    img_num = len(dic)
    if session_num is not None:
        img_num = 1000
    result = np.zeros((img_num, 3, height, width))
    for i, obj in enumerate(dic):
        img = cv2.imread(src_dir + obj['id'])
        result[i, :, :, :] = np.transpose(img, (2, 0, 1))

        if i % 1000 == 999 and i > 0:
            print('Color images {i}'.format(i=i))
            if session_num is not None:
                break

    return result


def read_gray(json_path, width, height, src_dir, session_num = None):
    """
    Read in colorful images like src
    :param json_path: pose json
    :param src_dir: image from
    :return: images in (num, h, w)
    """
    f = open(json_path, 'r')
    dic = json.load(f)
    img_num = len(dic)
    if session_num is not None:
        img_num = 1000
    result = np.zeros((img_num, height, width))
    for i, obj in enumerate(dic):
        img = cv2.imread(src_dir + obj['id'], 0)
        result[i, :, :] = img

        if i % 1000 == 999 and i > 0:
            print('Gray images {i}'.format(i=i))
            if session_num is not None:
                break

    return result


def walk_through_transformations(src_dir, match_json_path, data_json_path, mapping, dst_dir, num = 2):
    """
    walk through the top matches and output them into a folder
    :param match_json_path: match.json
    :param data_json_path: pose.json
    :param mapping: just mapping
    :return: none
    """
    f = open(match_json_path, 'r')
    match = json.load(f)
    f.close()
    f = open(data_json_path, 'r')
    data = json.load(f)
    f.close()

    limbs = np.array([[1, 2, 3, 4], [5, 7], [6, 8], [7, 9], [8, 10], [11, 13],
                      [12, 14], [13, 15], [14, 16], [5, 6, 11, 12]])

    img_num = len(data)
    for i, sample in enumerate(data):
        matcher = match[sample['id']]
        print(i)
        for j in range(num):
            src_name = matcher[j]['id']
            src_img = cv2.imread(src_dir + src_name)
            joints0 = data[mapping[src_name]]['joints']
            joints1 = sample['joints']

            dst_img = move.warp_img_pair(limbs, joints0, joints1, src_img)
            dst_path = dst_dir + sample['id'][:-4] + str(j) + '.jpg'
            cv2.imwrite(dst_path, dst_img)


def read_morphed_img_and_group(src_dir, data_json_path, match_json_path, width, height, num = 2, session_num = None):
    """
    walk through the whole transformed images and read in
    """
    f = open(match_json_path, 'r')
    match = json.load(f)
    f.close()
    f = open(data_json_path, 'r')
    data = json.load(f)
    f.close()

    img_num = len(data)
    if session_num is not None:
        img_num = 1000
    imgs = np.zeros((img_num, num, 3, height, width))

    for i, sample in enumerate(data):
        matcher = match[sample['id']]
        for j in range(num):
            src_name = src_dir + sample['id'][:-4] + str(j) + '.jpg'
            imgs[i, j, :, :, :] = np.transpose(cv2.imread(src_name), (2 ,0, 1))

        if i % 1000 == 999 and i > 0:
            print('Read morph {i}'.format(i = i))
            if session_num is not None:
                break

    return imgs

if __name__ == '__main__':
    mapping = construct_mapping('/home/pzq/warp_data/pose/market/market-final.json')
    walk_through_transformations('/home/pzq/warp_data/img/market/', '/home/pzq/warp_data/pose/market/match.json',
                                 '/home/pzq/warp_data/pose/market/market-final.json', mapping, '/home/pzq/warp_data/img/market_warped/')
