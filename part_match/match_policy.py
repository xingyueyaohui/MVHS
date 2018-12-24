"""
This is the file for matching policy
Including:
   Naively computing Euclidean loss
"""
import numpy as np
import dataset

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
        mid.append({'id': i, 'delta': delta})

    mid = sorted(mid, key = lambda x:x['delta'])

    if len(mid) >= num:
        return mid[:num]
    else:
        return mid


