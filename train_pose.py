"""
In the first stage, use PNGAN for refinement
"""
from model.poseNet import poseNet as poseNet
from model.PNNet import Res_Generator as Generator
from model.refine_block import Refiner as Refiner
from model.DC_descriminator import Discriminator as Discriminator
from lib.dataset import Dataset, Person
import lib.move_util as move
import lib.data_generator as dg
import lib.prework as pre

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
import numpy as np
import numpy.random as rand
import pickle
import json
import os

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

def run():
    """
    run the model
    """
    epoch = 1
    batch_size = 40
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    lr = 5 * 1e-4

    generator = Generator(64, 6).cuda()
    generator.load_state_dict(torch.load('/home/dongkai/pzq/MVHS/pretrain/PNG_6_s4e9.pkl'))
    refiner = Refiner(16)
    refiner.load_state_dict(torch.load('path'))

    pose_net = poseNet(2 * 17, 10)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas = (0.5, 0.999))

    f = open('/home/dongkai/pzq/MVHS/data/joint_tgt.json')
    joints_tgt = json.load(f)
    f = open('/home/dongkai/pzq/MVHS/data/joint_src.json')
    joints_src = json.load(f)
    f = open('/home/dongkai/pzq/MVHS/data/score.json')
    score = json.load(f)
    joints = joints_tgt - joints_src

    [num, _, _] = joints_tgt.shape
    num = int((num // 2000) * 2000)
    batch_num = num // batch_size

    for epo in range(epoch):
       for batch_index in range(batch_num):
            input = joints[batch_num * batch_size:, (batch_num+1) * batch_size]
            target = score[batch_num * batch_size:, (batch_num+1) * batch_size]
            input = np.view(input, 2*17)
            input = torch.tensor(input).float().cuda()
            target = torch.tensor(target).float().cuda()

            out = pose_net(input)
            optimizer.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            print('Loss for batch [{batch_num}] is [{loss}]'.format(batch_num=batch_num, loss=loss.detach().cpu()))

    torch.save(generator.state_dict(), '/home/dongkai/pzq/MVHS/pretrain/Pose.pkl')


def load_pose(data_json_path, joints_num = 17):
    """
    load pose data from data json
    """
    f = open(data_json_path)
    data = json.load(f)
    img_num = len(data)
    joints = np.arrays(img_num, joints_num, 2)
    for img in data:
        joints[img, :, :] = img['joints']

    return joints


def compute_pose_loss(img_src, img_tgt, joint_tgt):
    score = np.zeros(10)
    mask = np.where(move.make_limb_masks(limbs, joint_tgt, 64, 128) > 0.3, 1, 0)
    img_diff = np.abs(img_src - img_tgt)
    img_diff = np.sum(img_diff, dim=1)

    for i in range(10):
        score[i] = np.sum(np.view(img_diff * mask, -1))
    score = score / np.sum(score)

    return score


def compute_morph_distance(src_dir, morph_dir, data_json_path, match_json_path, width, height, mapping, joints_num=17, part_num=10, num=2):
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
    for i, sample in enumerate(data):
        if len(match[sample['id']] < num):
            img_num -= 1

    joints_src = np.zeros((2 * img_num, joints_num, 2))
    joints_tgt = np.zeros((2 * img_num, joints_num, 2))
    score = np.zeros((2 * img_num, part_num))

    for i, sample in enumerate(data):
        matcher = match[sample['id']]
        if (len(matcher) < 2):
            continue
        for j in range(num):
            joints_tgt[2 * i + j, :, :] = np.array(sample['joints'])
            joints_src[2 * i + j, :, :] = np.array(data[mapping[matcher[j]['id']]]['joints'])
            src_name = morph_dir + sample['id'][:-4] + str(j) + '.jpg'
            img = np.transpose(cv2.imread(src_name), (2 ,0, 1))
            tgt = np.transpose(cv2.imread(src_dir + sample['id']), (2, 0, 1))
            score[2 * i + j, :] = compute_pose_loss(img, tgt, np.array(sample['jonits']))

        if i % 1000 == 999 and i > 0:
            print('Read morph {i}'.format(i = i))

    f = open('/home/dongkai/pzq/MVHS/data/joint_tgt.json')
    json.dump(joints_tgt.tolist(), f)
    f = open('/home/dongkai/pzq/MVHS/data/joint_src.json')
    json.dump(joints_src.tolist(), f)
    f = open('/home/dongkai/pzq/MVHS/data/score.json')
    json.dump(score.tolist(), f)


if __name__ == '__main__':
    run()
