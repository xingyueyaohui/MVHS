"""
This is the file for original version
Containing the following things:
    refine the network
    (using pose-network to learn the final difference)
"""
import model.UNet as UNet
import model.poseNet as poseNet
from lib.dataset import Dataset, Person
import lib.move_util as move
import lib.data_generator as dg

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
    epoch = 100
    batch_size = 40
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    lr = 1e-3

    unet = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(unet.parameters(), lr, momentum = 0.05)

    height = 128
    width = 64

    dataset = pickle.load(open('/home/pzq/warp_data/pose/market/dataset.pkl', 'rb'))
    src_dir = '/home/pzq/warp_data/img/market/'
    pose_map_dir = '/home/pzq/warp_data/img/market_pose_map/'
    tgt_mask_dir = '/home/pzq/warp_data/img/market_tgt_mask/'
    match_file = open('/home/pzq/warp_data/pose/market/match.json')
    match = json.load(match_file)

    print("Loading Done")

    instance_num = 4   # 4 people warping for one id
    batch_people_num = batch_size / instance_num # people num in batch
    batch_num = int(dataset.people_num / batch_people_num) # batch number in an epoch

    loss_change = []

    for epo in range(epoch):
        people_id = 0
        total_loss = 0

        for batch_id in range(batch_num):
            # prepare the data
            batch_in = np.zeros((batch_size, height, width, 4))
            batch_gt = np.zeros((batch_size, height, width, 3))

            for i in range(batch_people_num):
                person = dataset.people[people_id]
                people_id += 1
                person_img_num = person.img_num
                tgt = rand.randint(0, person_img_num)
                tgt_img_name = person.imgs[tgt]
                joints_tgt = person.joints[tgt]
                tgt_img = cv2.imread(src_dir + tgt_img_name)
                tgt_pose = cv2.imread(pose_map_dir + tgt_img_name, 0)
                match_imgs = match[tgt_img_name]

                for j in range(instance_num):
                    batch_gt[i * instance_num + j, :, :, :] = tgt_img
                    batch_in[i * instance_num + j, :, :, 3] = tgt_pose
                    input_img = cv2.imread(src_dir + match_imgs[j]['id'])
                    joints_src = person.joints[match_imgs[j]['index']]
                    batch_in[i * instance_num + j, :, :, :2] = move.warp_img_pair(limbs, joints_src, joints_tgt, input_img)

            # Start training
            optimizer.zero_grad()
            out = unet(batch_in)
            loss = criterion(out, batch_gt)
            loss.backward()
            optimizer.step()

            print('Batch [{i}] in epoch [{e}], loss is [{loss}]'.format(i=batch_id, e=epo, loss=loss))
            total_loss += loss
        loss_change.append(total_loss)
        print('Epoch [{e}] has loss of [{loss}]'.format(e = epo, loss = total_loss))
        np.savetxt('loss_change.txt', np.array(loss_change))

if __name__ == '__main__':
    run()