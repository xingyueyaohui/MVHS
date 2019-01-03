"""
This is the file for original version
Containing the following things:
    refine the network
    (using pose-network to learn the final difference)
"""
from model.UNet import UNet as UNet
from model.poseNet import poseNet as poseNet
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
    epoch = 50
    batch_size = 40
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    lr = 5 * 1e-4
    session = 5

    unet = UNet(useBN=True).cuda()
    unet.load_state_dict(torch.load('/home/pzq/pose_warp/trained_model/bg_s4e47.pkl'))
    criterion = nn.L1Loss()
    advLoss = nn.BCELoss()
    optimizer = optim.SGD(unet.parameters(), lr, momentum = 0.2)

    height = 128
    width = 64

    dataset = pickle.load(open('/home/pzq/warp_data/pose/market/dataset.pkl', 'rb'))
    src_dir = '/home/pzq/warp_data/img/market/'
    pose_map_dir = '/home/pzq/warp_data/img/market_pose_map/'
    tgt_mask_dir = '/home/pzq/warp_data/img/market_tgt_mask/'
    warp_dir = '/home/pzq/warp_data/img/market_warped/'
    match_path = '/home/pzq/warp_data/pose/market/match.json'
    match_file = open(match_path, 'r')
    data_json_path = '/home/pzq/warp_data/pose/market/market-final.json'
    data_file = open(data_json_path, 'r')
    match = json.load(match_file)
    data = json.load(data_file)

    print("Loading Done. Start perquisite job.")
    mapping = pre.construct_mapping(data_json_path)
    src_imgs = pre.read_color(data_json_path, width, height, src_dir) / 255
    print('Read in src images done')
    pose_map = pre.read_gray(data_json_path, width, height, pose_map_dir) / 255
    tgt_masks = pre.read_gray(data_json_path, width, height, tgt_mask_dir) / 255
    print('Gray images reading done')
    warped_imgs = pre.read_morphed_img_and_group(warp_dir, data_json_path, match_path, width, height) / 255
    print('Warped images reading done')

    img_num = len(data)
    train_list = np.arange(0, img_num)

    instance_num = 2   # 4 people warping for one id
    batch_img_num = int(batch_size / instance_num) # people num in batch
    batch_num = int(img_num / batch_img_num) # batch number in an epoch

    loss_change = []

    for epo in range(epoch):
        people_id = 0
        total_loss = 0
        np.random.shuffle(train_list)

        for batch_id in range(batch_num):
            # prepare the data
            batch_in = np.zeros((batch_size, 4, height, width))
            batch_gt = np.zeros((batch_size, 3, height, width))
            batch_mask = np.zeros((batch_size, height, width))
            #print("Start prepare")
            for i in range(batch_img_num):
                index = train_list[batch_img_num * batch_id + i]
                tgt_img = src_imgs[index]
                tgt_pose = pose_map[index]
                tgt_mask = tgt_masks[index]

                for j in range(instance_num):
                    batch_gt[instance_num * i + j, :, :, :] = tgt_img
                    batch_in[instance_num * i + j, 3, :, :] = tgt_pose
                    batch_in[instance_num * i + j, :3, :, :] = warped_imgs[index, j, :, :, :] * tgt_mask \
                                                               + (1 - tgt_mask) * tgt_img
                    batch_mask[instance_num * i + j, :, :] = tgt_mask

            #print("End of prepare")

            # Start training
            optimizer.zero_grad()
            batch_in = torch.tensor(batch_in).float().cuda()
            batch_gt = torch.tensor(batch_gt).float().cuda()
            out = unet(batch_in)
            loss = criterion(out, batch_gt)
            loss.backward()
            optimizer.step()

            print('Batch [{i}] in epoch [{e}], loss is [{loss}]'.format(i=batch_id, e=epo, loss=loss.detach().cpu()))
            total_loss += loss

        loss_change.append(total_loss)
        print('Epoch [{e}] has loss of [{loss}]'.format(e = epo, loss = total_loss))
        np.savetxt('loss_change{s}.txt'.format(s=session), np.array(loss_change))

        if (epo + 1) % 5 == 0:
            torch.save(unet.state_dict(), '/home/pzq/pose_warp/pretrain/s{s}e{epo}.pkl'.format(s=session,epo=epo))


def test_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    unet = UNet(useBN=True).cuda()
    unet.load_state_dict(torch.load('/home/pzq/pose_warp/pretrain/s5e49.pkl'))

    width = 64
    height = 128

    src_dir = '/home/pzq/warp_data/img/market/'
    pose_map_dir = '/home/pzq/warp_data/img/market_pose_map/'
    tgt_mask_dir = '/home/pzq/warp_data/img/market_tgt_mask/'
    warp_dir = '/home/pzq/warp_data/img/market_warped/'
    match_path = '/home/pzq/warp_data/pose/market/match.json'
    match_file = open(match_path, 'r')
    data_json_path = '/home/pzq/warp_data/pose/market/market-final.json'
    data_file = open(data_json_path, 'r')
    match = json.load(match_file)
    data = json.load(data_file)

    print("Loading Done. Start perquisite job.")
    mapping = pre.construct_mapping(data_json_path)
    src_imgs = pre.read_color(data_json_path, width, height, src_dir, session_num=1000) / 255
    print('Read in src images done')
    pose_map = pre.read_gray(data_json_path, width, height, pose_map_dir, session_num=1000) / 255
    tgt_masks = pre.read_gray(data_json_path, width, height, tgt_mask_dir, session_num=1000) / 255
    print('Gray images reading done')
    warped_imgs = pre.read_morphed_img_and_group(warp_dir, data_json_path, match_path, width, height, num=2, session_num=1000) / 255
    print('Warped images reading done')

    img_num = 1000
    train_list = np.arange(0, img_num)
    batch_size = 40
    instance_num = 2  # 4 people warping for one id
    batch_img_num = int(batch_size / instance_num)  # people num in batch
    batch_num = int(img_num / batch_img_num)  # batch number in an epoch

    np.random.shuffle(train_list)

    # prepare the data
    batch_in = np.zeros((batch_size, 4, height, width))
    batch_gt = np.zeros((batch_size, 3, height, width))
    batch_mask = np.zeros((batch_size, height, width))
    # print("Start prepare")
    for i in range(batch_img_num):
        index = train_list[i]
        tgt_img = src_imgs[index]
        tgt_pose = pose_map[index]
        tgt_mask = tgt_masks[index]

        for j in range(instance_num):
            batch_gt[instance_num * i + j, :, :, :] = tgt_img
            batch_in[instance_num * i + j, 3, :, :] = tgt_pose
            #batch_in[instance_num * i + j, :3, :, :] = warped_imgs[index, j, :, :, :] * tgt_mask \
            #                                           + (1 - tgt_mask) * tgt_img
            batch_mask[instance_num * i + j, :, :] = tgt_mask
            batch_in[instance_num * i + j, :3, :, :] = warped_imgs[index, j, :, :, :]
    # print("End of prepare")

    # Start testing
    batch_in = torch.tensor(batch_in).float().cuda()
    with torch.no_grad():
        out = unet(batch_in)

    out = (np.array(out.detach().cpu()) * 255).astype(np.uint8)
    batch_gt = (batch_gt * 255).astype(np.uint8)
    batch_in = (np.array(batch_in.detach().cpu()) * 255).astype(np.uint8)

    # Output the results
    out_dir = '/home/pzq/pose_warp/vis_data/'
    for i in range(batch_size):
        img_syn = np.transpose(out[i], (1, 2, 0))
        img_tgt = np.transpose(batch_gt[i], (1, 2, 0))
        img_src = np.transpose(batch_in[i], (1, 2, 0))
        cv2.imwrite(out_dir + 's' + str(i).zfill(2) + '.jpg', img_syn)
        cv2.imwrite(out_dir + 't' + str(i).zfill(2) + '.jpg', img_tgt)
        cv2.imwrite(out_dir + 'o' + str(i).zfill(2) + '.jpg', img_src)


if __name__ == '__main__':
    run()
    # test_model()

