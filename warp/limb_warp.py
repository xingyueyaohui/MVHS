"""
File for warping limbs operations
"""


import numpy as np
import cv2

"""
limb index in alpha-pose format
This is the format of the key points
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
"""
limb = np.array([[13, 17, 18, 12], [9, 10], [8, 7], [10, 11], [7, 6], [3, 4],
                 [2, 1], [4, 5], [1, 0], [9, 8, 3, 2]])
limb_name = ['head', 'upper-left-arm', 'upper-right-arm', 'down-left-arm',
             'down-right-arm', 'upper-left-leg', 'upper-right-leg',
             'down-left-leg', 'down-right-leg', 'torso']


"""
Make gaussian map according to center points and variance
"""
def make_gaussian_map(height, width, center, var_x, var_y, theta):
    xv, yv = np.meshgrid(np.array(range(width)), np.array(range(height)),
                         sparse=False, indexing='xy')
    a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
    b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
    c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

    return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                    2 * b * (xv - center[0]) * (yv - center[1]) +
                    c * (yv - center[1]) * (yv - center[1])))


"""
Make limb masks
"""
def make_limb_masks(limbs, joints, height, width):
    n_limbs = len(limbs)
    mask = np.zeros((height, width, n_limbs))

    # Gaussian sigma perpendicular to limb axis
    sigma_prep = np.array([12, 12, 12, 12, 12, 12, 12, 12, 12, 13]) ** 2

    for i in range(n_limbs):
        n_joints_for_limb = len(limbs[i])
        p = np.zeros((n_joints_for_limb, 2))

        for j in range(n_joints_for_limb):
            p[j, 0] = joints[limbs[i, j], 0]
            p[j, 1] = joints[limbs[i, j], 1]

        if n_joints_for_limb == 4:
            p_top = np.mean(p[0:2, :], axis=0)
            p_bot = np.mean(p[2:4, :], axis=0)
            p = np.vstack((p_top, p_bot))

        center = np.mean(p, axis=0)
        sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 1.5])
        theta = np.artan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

        mask_i = make_gaussian_map(height, width, center, sigma_parallel, sigma_prep[i], theta)
        mask[:, :, i] = mask_i / (np.amax(mask_i) + 1e-6)

    return mask


"""
Warp image from info0 to info1
Possible modification:
    the way to get transformation matrix
"""
def warp_image(info0, info1):
    img0 = info0['img']
    img1 = info1['img']
    joints0 = info0['joints']
    joints1 = info1['joints']

    [height, width] = img1.shape
    sigma_joint = 7/4.0
    n_limbs = 10
    n_joints = 17

    x_mask_src = np.zeros((height, width, n_limbs))
    x_pose_src = np.zeros((height, width, n_joints))
    x_pose_tgt = np.zeros((height, width, n_joints))
    y = np.zeros((height, width, 3))

    x_mask_src = make_limb_masks(limb, joints0, height, width)
    x_mask_tgt = np.zeros((height, width, n_limbs))
    rot_matrix = []

    for i in range(n_limbs):
        key_point_index = limb[i]
        point_num = len(key_point_index)

        if point_num == 2:
            joint00 = joints0[key_point_index[0]]
            joint01 = joints0[key_point_index[1]]
            # make square due to the two points
            joint02 = [(joint00[0] + joint01[0] + joint00[1] - joint01[1]) / 2,
                       (joint00[1] + joint01[1] + joint01[0] - joint00[0]) / 2]
            joint03 = [(joint00[0] + joint01[0] - joint00[1] + joint01[1]) / 2,
                       (joint00[1] + joint01[1] - joint01[0] + joint00[0]) / 2]
            key_point0 = np.array([joint00, joint01, joint02, joint03]).astype(np.float32)

            joint10 = joints1[key_point_index[0]]
            joint11 = joints0[key_point_index[1]]
            # make square due to the two points
            joint12 = [(joint10[0] + joint11[0] + joint10[1] - joint11[1]) / 2,
                       (joint10[1] + joint11[1] + joint11[0] - joint10[0]) / 2]
            joint13 = [(joint10[0] + joint11[0] - joint10[1] + joint11[1]) / 2,
                       (joint10[1] + joint11[1] - joint11[0] + joint10[0]) / 2]
            key_point1 = np.array([joint10, joint11, joint12, joint13]).astype(np.float32)

            M = cv2.getPerspectiveTransform(key_point0, key_point1)
            rot_matrix.append(M)

        elif point_num == 4:
            key_point0 = []
            key_point1 = []
            for j in range(point_num):
                key_point1.append(joints1[key_point_index[j]])
                key_point0.append(joints0[key_point_index[j]])
            key_point0 = np.array(key_point0).astype(np.float32)
            key_point1 = np.array(key_point1).astype(np.float32)

            M = cv2.getPerspectiveTransform(key_point0, key_point1)
            rot_matrix.append(M)

        for i in range(n_limbs):
            for c in range(3):
                rot_part = x_mask_src[:, :, i] * img0[:, :, c]
                rot_part = cv2.warpPerspective(rot_part, rot_matrix[i], (width, height))
                x_mask_tgt[:, :, i] = cv2.warpPerspective(x_mask_src[:, :, i], rot_matrix[i], (width, height))
                y[:, :, c] = y[:, :, c] * (1 - x_mask_tgt[:, :, i]) + x_mask_tgt[:, :, i] * rot_part

        return y






