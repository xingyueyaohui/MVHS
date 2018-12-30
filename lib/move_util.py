"""
Tools for part moving
"""
import numpy as np
import lib.transformations as transformations
import cv2
import os


def make_limb_masks(limbs, joints, img_width, img_height):
    """
    make masks according to limb coordinates
    """
    n_limbs = len(limbs)
    mask = np.zeros((img_height, img_width, n_limbs))
    joints = np.array(joints)

    # Gaussian sigma perpendicular to the limb axis.
    sigma_perp = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 15]) ** 2

    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p[j, :] = [joints[limbs[i][j], 0], joints[limbs[i][j], 1]]

        if n_joints_for_limb == 4:
            p_top = np.mean(p[0:2, :], axis=0)
            p_bot = np.mean(p[2:4, :], axis=0)
            p = np.vstack((p_top, p_bot))

        center = np.mean(p, axis=0)

        sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 1.5])
        theta = np.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

        mask_i = make_gaussian_map(img_width, img_height, center, sigma_parallel, sigma_perp[i], theta)
        mask[:, :, i] = mask_i / (np.amax(mask_i) + 1e-6)

    return mask


def make_gaussian_map(img_width, img_height, center, var_x, var_y, theta):
    xv, yv = np.meshgrid(np.array(range(img_width)), np.array(range(img_height)),
                         sparse=False, indexing='xy')

    a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
    b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
    c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

    return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                    2 * b * (xv - center[0]) * (yv - center[1]) +
                    c * (yv - center[1]) * (yv - center[1])))


def get_person_transformations(limbs, joints0, joints1):
    """
    Calculate limb transformation according to
    similarity transform in transformations.py
    This is the function for the whole person
    :param limbs: same as limbs up there
    :param joints0: src
    :param joints1: dst
    :return Ms: rotation matrix with size (2, 3, n_limbs)
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

        tform = transformations.make_similarity(p0, p1)
        Ms[:, :, i] = np.array([[tform[1], -tform[3], tform[0]], [tform[3], tform[1], tform[2]]])

    return Ms


def get_limb_transformations(limbs, index0, index1, joints0, joints1):
    """
    Calculate limb transformations for a seperate limb
    :param limbs: limb joints
    :param limb_index: which limb
    :param joints0: src
    :param joints1: dst
    :return: rotation matrix of size (2, 3)
    """
    n_joints_for_limb = len(limbs[index0])
    p0 = np.zeros((n_joints_for_limb, 2))
    p1 = np.zeros((n_joints_for_limb, 2))
    index0 = int(index0)
    index1 = int(index1)
    for j in range(n_joints_for_limb):
        a = np.array([joints0[limbs[index0][j], 0], joints0[limbs[index0][j], 1]])
        p0[j, :] = np.array([joints0[limbs[index0][j], 0], joints0[limbs[index0][j], 1]])
        p1[j, :] = np.array([joints1[limbs[index1][j], 0], joints1[limbs[index1][j], 1]])

    tform = transformations.make_similarity(p0, p1)
    Ms = np.array([[tform[1], -tform[3], tform[0]], [tform[3], tform[1], tform[2]]])

    return Ms


def warp_img_pair(limbs, info0, info1, src_img):
    """
    Wrapper for simple part moving, from info0 to info1
    Use the easiest method as an example, contrast to warp_combine below
    info contains img name and joints
    :param info0: src person
    :param info1: tgt person
    """
    n_limbs = len(limbs)
    img_name0 = info0['id']
    img_name1 = info1['id']

    # print("From {id0} to {id1}".format(id0=img_name0, id1=img_name1))

    # each joint has two value, 0 for width and 1 for height
    joints0 = np.array(info0['joints'])
    joints1 = np.array(info1['joints'])

    img0 = src_img

    [height, width, _] = img0.shape

    rot_matrix = np.zeros((2, 3, n_limbs))

    src_limb_masks = make_limb_masks(limbs, joints0, width, height)
    tgt_limb_masks = make_limb_masks(limbs, joints1, width, height)
    rot_matrix = get_limb_transformations(limbs, joints0, joints1)

    img = np.zeros((height, width, 3))

    for i in range(0, n_limbs):

        rot = rot_matrix[:, :, i]
        mask = src_limb_masks[:, :, i]
        img_warped = cv2.warpAffine(img0, rot, (width, height))
        src_mask = cv2.warpAffine(src_limb_masks[:, :, i], rot, (width, height))
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - tgt_limb_masks[:, :, i]) + img_warped[:, :, c] * tgt_limb_masks[:, :, i]

    return img
