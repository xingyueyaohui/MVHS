'''
File for part moving
Default alpha-pose format
'''

import numpy as np
import os
import cv2
import transformations
import util
import json

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
                 [12, 14], [13, 15], [14, 16], [5, 6, 7, 8]])
limb_name = ['head', 'upper-left-arm', 'upper-right-arm', 'down-left-arm',
             'down-right-arm', 'upper-left-leg', 'upper-right-leg',
             'down-left-leg', 'down-right-leg', 'torso']


def warp_img_pair(info0, info1, src_dir, dst_dir):
    """
    Wrapper for simple part moving, from info0 to info1
    Use the easiest method as an example, contrast to warp_combine below
    :param info0: src person
    :param info1: tgt person
    """
    n_limbs = len(limbs)
    img_name0 = info0['id']
    img_name1 = info1['id']
    img_path0 = os.path.join(src_dir, img_name0)
    img_path1 = os.path.join(src_dir, img_name1)
    dst_path = dst_dir + 's' + img_name1

    # each joint has two value, 0 for width and 1 for height
    joints0 = np.array(info0['joints'])
    joints1 = np.array(info1['joints'])

    img0 = cv2.imread(img_path0)
    img1 = cv2.imread(img_path1)

    [height, width, _] = img0.shape

    rot_matrix = np.zeros((2, 3, n_limbs))

    src_limb_masks = util.make_limb_masks(limbs, joints0, width, height)
    tgt_limb_masks = util.make_limb_masks(limbs, joints1, width, height)
    rot_matrix = get_limb_transformations(limbs, joints0, joints1)

    img = np.zeros((height, width, 3))

    for i in range(n_limbs):
        rot = rot_matrix[:, :, i]
        mask = src_limb_masks[:, :, i]
        img_warped = cv2.warpAffine(img0, rot, (width, height))
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - tgt_limb_masks[:, :, i]) + img_warped[:, :, c] * tgt_limb_masks[:, :, i]

    cv2.imwrite(dst_path, img)


def get_limb_transformations(limbs, joints0, joints1):
    """
    Calculate limb transformation according to
    similarity transform in transformations.py
    :param limbs: same as limbs up there
    :param joints0: src
    :param joints1: dst
    :return Ms: rotation matrix
    """
    n_limbs = len(limbs)
    Ms = np.zeros((2, 3, n_limbs))
    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p0 = np.zeros((n_joints_for_limb, 2))
        p1 = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p0[j, :] = [joints0[limbs[i][j], 0], joints0[limbs[i][j], 1]]
            p1[j, :] = [joints1[limbs[i][j], 0], joints1[limbs[i][j], 1]]

        tform = transformations.make_similarity(p1, p0)
        Ms[:, :, i] = np.array([[tform[1], -tform[3], tform[0]], [tform[3], tform[1], tform[2]]])

    return Ms


def morphing(origin_pose, target_pose, origin_img, target_img, limbs):
    """
    Most careful part moving
    :param target_image should be input, as this may start with a half-complete image
    :return tgt image
    """
    n_limbs = len(limbs)
    [height, width, _] = target_img.shape
    target_size = [width, height, 3]
    target_mask = np.zeros((height, width, n_limbs))
    origin_mask = np.zeros((height, width, n_limbs))
    target_mask = util.make_limb_masks(limbs, target_pose, width, height)
    origin_mask = util.make_limb_masks(limbs, origin_pose, width, height)

    for i, limb in enumerate(limbs):

        # Better at
        if len(limb) != 2:
            continue
        origin_body_part = origin_img

        origin_pose_a = origin_pose[limb[0]]
        origin_pose_b = origin_pose[limb[1]]
        origin_pose_part = origin_pose_b - origin_pose_a
        target_pose_a = target_pose[limb[0]]
        target_pose_b = target_pose[limb[1]]
        target_pose_part = target_pose_b - target_pose_a
        origin_part_len = np.sqrt(np.sum(np.square(origin_pose_part)))
        target_part_len = np.sqrt(np.sum(np.square(target_pose_part)))

        # scaling here
        scale_factor = target_part_len / origin_part_len
        if scale_factor == 0:
            continue

        # rotating angle, in form of x/y
        theta = - (np.arctan2(target_pose_part[0], target_pose_part[1]) - np.arctan2(
            origin_pose_part[0], origin_pose_part[1])) * 180 / np.pi

        # scale
        origin_size = [width, height, 3]
        origin_size[0] *= scale_factor
        origin_size[1] *= scale_factor
        origin_pose_a *= scale_factor
        origin_pose_b *= scale_factor
        origin_body_part = cv2.resize(origin_body_part, (int(origin_size[0]), int(origin_size[1])),
                                      interpolation=cv2.INTER_CUBIC)

        # translate to the center
        origin_pose_center = (origin_pose_a + origin_pose_b) / 2
        origin_center = [int(origin_size[0] / 2), int(origin_size[1] / 2)]
        [tx, ty] = origin_center - origin_pose_center
        tm = np.float32([[1, 0, tx], [0, 1, ty]])
        origin_body_part = cv2.warpAffine(origin_body_part, tm, (int(origin_size[0]), int(origin_size[1])))

        # rotate
        rm = cv2.getRotationMatrix2D((origin_center[0], origin_center[1]), theta, 1)
        origin_body_part = cv2.warpAffine(origin_body_part, rm, (int(origin_size[0]), int(origin_size[1])))

        # crop and paste
        # find the area first
        # then crop
        target_pose_center = (target_pose_a + target_pose_b) / 2
        target_pose_center[0] = int(target_pose_center[0])
        target_pose_center[1] = int(target_pose_center[1])
        if target_pose_center[1] >= origin_center[1]:
            origin_row_low = 0
            target_row_low = target_pose_center[1] - origin_center[1]
        else:
            origin_row_low = origin_center[1] - target_pose_center[1]
            target_row_low = 0
        if (target_size[1] - target_pose_center[1]) >= (origin_size[1] - origin_center[1]):
            origin_row_high = origin_size[1]
            target_row_high = target_pose_center[1] + origin_size[1] - origin_center[1]
        else:
            origin_row_high = origin_center[1] + target_size[1] - target_pose_center[1]
            target_row_high = target_size[1]
        if target_pose_center[0] >= origin_center[0]:
            origin_col_low = 0
            target_col_low = target_pose_center[0] - origin_center[0]
        else:
            origin_col_low = origin_center[0] - target_pose_center[0]
            target_col_low = 0
        if (target_size[0] - target_pose_center[0]) >= (origin_size[0] - origin_center[0]):
            origin_col_high = origin_size[0]
            target_col_high = target_pose_center[0] + origin_size[0] - origin_center[0]
        else:
            origin_col_high = origin_center[0] + target_size[0] - target_pose_center[0]
            target_col_high = target_size[0]

        origin_row_low = int(origin_row_low)
        target_row_low = int(target_row_low)
        origin_row_high = int(origin_row_high)
        target_row_high = int(target_row_high)
        origin_col_low = int(origin_col_low)
        target_col_low = int(target_col_low)
        origin_col_high = int(origin_col_high)
        target_col_high = int(target_col_high)

        target_part = target_img[target_row_low:target_row_high, target_col_low:target_col_high, :]
        origin_part = origin_body_part[origin_row_low:origin_row_high, origin_col_low:origin_col_high, :]
        target_part_mask = target_mask[target_row_low:target_row_high, target_col_low:target_col_high, i]
        for c in range(3):
            target_part[:,:,c] = target_part[:,:,c] * (1 - target_part_mask) + \
                                 origin_part[:,:,c] * target_part_mask
        target_img[target_row_low:target_row_high, target_col_low:target_col_high, :] = target_part

    return target_img

def warp_combine(info0, info1, src_dir, dst_dir):
    """
    An example for more complex part moving
    """
    n_limbs = len(limbs)
    img_name0 = info0['id']
    img_name1 = info1['id']
    img_path0 = os.path.join(src_dir, img_name0)
    img_path1 = os.path.join(src_dir, img_name1)
    dst_path = dst_dir + 'sc' + img_name1

    # each joint has two value, 0 for width and 1 for height
    joints0 = np.array(info0['joints'])
    joints1 = np.array(info1['joints'])

    img0 = cv2.imread(img_path0)
    img1 = cv2.imread(img_path1)

    [height, width, _] = img0.shape

    rot_matrix = np.zeros((2, 3, n_limbs))

    src_limb_masks = util.make_limb_masks(limbs, joints0, width, height)
    tgt_limb_masks = util.make_limb_masks(limbs, joints1, width, height)
    rot_matrix = get_limb_transformations(limbs, joints0, joints1)

    img = np.zeros((height, width, 3))

    # First the limbs
    img = morphing(joints0, joints1, img0, img, limbs)

    # Then the part with more than 2 key points like torso
    for i in range(n_limbs):
        limb = limbs[i]
        if len(limb) == 4:
            rot = rot_matrix[:, :, i]
            mask = src_limb_masks[:, :, i]
            img_warped = cv2.warpAffine(img0, rot, (width, height))
            for c in range(3):
                img[:, :, c] = img[:, :, c] * (1 - tgt_limb_masks[:, :, i]) + img_warped[:, :, c] * tgt_limb_masks[:, :, i]

    cv2.imwrite(dst_path, img)


if __name__ == '__main__':
    """
    Example generation
    """
    test_pairs = [[1, 2], [72, 73]] # test two pairs as an example
    file = open('pose.json', 'r')
    people = json.load(file)
    for pair in test_pairs:
        person0 = people[pair[0]]
        person1 = people[pair[1]]
        warp_combine(person0, person1, '../data/gt_bbox/', 'vis_data/',)
        warp_combine(person1, person0, '../data/gt_bbox/', 'vis_data/',)