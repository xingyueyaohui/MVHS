"""
This is the file for generating data for training
Including:
    draw joints heat map as input
    euclidean match
    store the matching result into json
    draw skeleton as input -- to be done
"""
import json
import cv2
import numpy as np
import pickle
from lib.dataset import Dataset, Person
import lib.move_util as move

"""
    (17 body parts)
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
"""
limbs = np.array([[1, 2, 3, 4], [5, 7], [6, 8], [7, 9], [8, 10], [11, 13],
                 [12, 14], [13, 15], [14, 16], [5, 6, 11, 12]])
limb_name = ['head', 'upper-left-arm', 'upper-right-arm', 'down-left-arm',
             'down-right-arm', 'upper-left-leg', 'upper-right-leg',
             'down-left-leg', 'down-right-leg', 'torso']

def draw_joint_heatmap(json_path, src_dir, dst_dir):
    """
    generating pose map
    """
    file = open(json_path, 'r')
    dic = json.load(file)

    for person in dic:
        img_name = person['id']
        img = cv2.imread(src_dir + img_name)
        [height, width, _] = img.shape
        map = np.zeros((height, width))

        key_points = np.array(person['joints'])
        joints_num = key_points.shape[0]
        for i in range(joints_num):
            x = int(key_points[i][0])
            y = int(key_points[i][1])
            cv2.circle(map, (x, y), 2, (255, 255, 255), thickness=-1)

        cv2.imwrite(dst_dir + img_name, map)

# As for the training stage
# We don't know which are the best match
# Not good enough basis misguides the network
# Therefore, we intuitively choose the best five for training
def normalize(joints, format='alpha'):
    """
    Normalize the joints location according to center of human
    :param joints: coordinates of joints
    :param format: which pose estimator
    :return: normalized pose
    """

    center = np.zeros(2)
    for i in limbs[9]:
        center += joints[i]
    center /= 4
    n_joints = joints.shape[1]
    new_joints = np.zeros((n_joints, 2))

    for i in range(n_joints):
        new_joints[i] = joints[i] - center

    return new_joints


def euclid_match_human(person, tgt, num = 5):
    """
    Find the top 5 matches of whole human
    :param person: information about this person
    :param tgt: target want to synthesize
    :param num: how many
    :return: top num matches
    """

    tgt_joint = np.array(person.joints[tgt])
    tgt_joint = normalize(tgt_joint)

    mid = []
    n_joints = tgt_joint.shape[0]
    n_imgs = len(person.imgs)
    for i in range(n_imgs):
        if i == tgt:
            continue
        spl_joint = person.joints[i]
        spl_joint = normalize(spl_joint)
        delta_joint = spl_joint - tgt_joint
        delta = np.sum(np.square(delta_joint.reshape(2 * n_joints)))
        mid.append({'index':i, 'id': person.imgs[i], 'delta': delta})

    mid = sorted(mid, key = lambda x:x['delta'])

    if len(mid) >= num:
        return mid[:num]
    else:
        return mid


def make_match_json(json_path, dst_dir, num = 4):
    """
    prepare the top 4 match for each id
    store the dataset in dst_dir and write the results into json
    """
    dataset = Dataset()
    dataset.make_data(json_path)
    pickle.dump(dataset, open(dst_dir + 'dataset.pkl', 'wb'))

    dic = {}
    for person in dataset.people:
        for i in range(person.img_num):
            match = euclid_match_human(person, i, num)
            dic[person.imgs[i]] = match

    f = open(dst_dir + 'match.json', 'w')
    json.dump(dic, f)


def make_mask(json_path, src_dir, dst_dir, limbs):
    """
    make the target mask for the convenience of training
    """
    f = open(json_path, 'r')
    dic = json.load(f)

    for person in dic:
        img_name = person['id']
        img_path = src_dir + img_name
        img = cv2.imread(img_path)
        [height, width, _] = img.shape
        mask = move.make_limb_masks(limbs, person['joints'], width, height)

        mask = np.sum(mask, axis = 2)
        mask = np.where(mask > 0.5, 255.0, 0.0)
        cv2.imwrite(dst_dir + img_name, mask)
