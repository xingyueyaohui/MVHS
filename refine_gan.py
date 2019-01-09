"""
This is the file for original version
Containing the following things:
    refine the network
    (using pose-network to learn the final difference)
"""
from model.UNet import UNet as UNet
from model.poseNet import poseNet as poseNet
from model.PNNet import Res_Generator as Generator
from model.DC_descriminator import Discriminator as Discriminator
from model.mask_loss import MaskLoss as MaskLoss
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
    epoch = 20
    batch_size = 40
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    lr_generator = 1.5 * 1e-4
    lr_discriminator = 1.00 * 1e-4
    session = 2
    lam = 10

    # generator = UNet(useBN=True).cuda()
    # generator.load_state_dict(torch.load('/home/pzq/pose_warp/pretrain/G_s1e14.pkl'))
    generator = Generator(64, 3).cuda()
    generator.load_state_dict(torch.load('/home/pzq/pose_warp/trained_model/PNG_base.pkl'))
    discriminator = Discriminator(128, 64).cuda()
    discriminator.load_state_dict(torch.load('/home/pzq/pose_warp/trained_model/PND_base.pkl'))

    criterion = nn.L1Loss()
    mask_loss = MaskLoss(criterion)
    advLoss = nn.BCELoss()
    G_optimizer = optim.Adam(generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))

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

    img_num = 20000
    train_list = np.arange(0, img_num)

    instance_num = 2   # 4 people warping for one id
    batch_img_num = int(batch_size / instance_num) # people num in batch
    batch_num = int(img_num / batch_img_num) # batch number in an epoch

    G_loss_change = []
    D_loss_change = []
    loss_change = []

    for epo in range(epoch):
        people_id = 0
        total_G_loss = 0
        total_D_loss = 0
        total_loss = 0
        np.random.shuffle(train_list)

        for batch_id in range(batch_num):
            # prepare the data
            batch_in = np.zeros((batch_size, 4, height, width))
            batch_gt = np.zeros((batch_size, 3, height, width))
            batch_mask = np.zeros((batch_size, height, width))
            batch_dis = np.zeros((batch_size, 3, height, width))
            #print("Start prepare")
            for i in range(batch_img_num):
                index = train_list[batch_img_num * batch_id + i]
                tgt_img = src_imgs[index]
                tgt_pose = pose_map[index]
                tgt_mask = tgt_masks[index]

                for j in range(instance_num):
                    batch_gt[instance_num * i + j, :, :, :] = tgt_img
                    batch_in[instance_num * i + j, 3, :, :] = tgt_pose
                    #batch_dis[instance_num * i + j, 3, :, :] = tgt_pose
                    batch_in[instance_num * i + j, :3, :, :] = warped_imgs[index, j, :, :, :] * tgt_mask \
                                                               + (1 - tgt_mask) * tgt_img
                    batch_mask[instance_num * i + j, :, :] = tgt_mask

            batch_dis[:, :, :, :] = batch_gt
            #print("End of prepare")

            # Start training
            G_optimizer.zero_grad()
            D_optimizer.zero_grad()

            batch_in = torch.tensor(batch_in).float().cuda()
            batch_dis = torch.tensor(batch_dis).float().cuda()
            batch_gt = torch.tensor(batch_gt).float().cuda()

            labels = np.zeros(2 * batch_size)
            labels[:batch_size] = 1
            labels = torch.from_numpy(labels.astype(np.float32)).cuda()

            out = generator(batch_in)
            dis_input = torch.cat([batch_dis, out])
            dis_out = discriminator(dis_input)

            gt_loss = criterion(out, batch_gt)
            G_loss = advLoss(dis_out[batch_size:, 0], labels[:batch_size])
            generator_loss = G_loss + lam * gt_loss
            generator_loss.backward(retain_graph=True)
            G_optimizer.step()

            D_loss = advLoss(dis_out[:, 0], labels)
            D_loss.backward()
            D_optimizer.step()

            G_loss = G_loss.detach().cpu()
            D_loss = D_loss.detach().cpu()
            gt_loss = gt_loss.detach().cpu()

            print('Batch [{i}] in epoch [{e}], distance is [{distance}], G_loss is [{G_loss}], D_loss is [{D_loss}]'.format(
                i=batch_id, e=epo, distance=gt_loss, G_loss=G_loss, D_loss=D_loss))
            total_G_loss += G_loss
            total_D_loss += D_loss
            total_loss += gt_loss

        G_loss_change.append(total_G_loss)
        D_loss_change.append(total_D_loss)
        loss_change.append(total_loss)
        np.savetxt('G_loss_change{s}.txt'.format(s=session), np.array(G_loss_change))
        np.savetxt('D_loss_change{s}.txt'.format(s=session), np.array(D_loss_change))
        np.savetxt('loss_change{s}.txt'.format(s=session), np.array(loss_change))

        if (epo + 1) % 5 == 0:
            torch.save(generator.state_dict(), '/home/pzq/pose_warp/pretrain/PNG_s{s}e{epo}.pkl'.format(s=session,epo=epo))
            torch.save(discriminator.state_dict(), '/home/pzq/pose_warp/pretrain/D_s{s}e{epo}.pkl'.format(s=session,epo=epo))


def out_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # generator = UNet(useBN=True).cuda()
    generator = Generator(64, 3).cuda()
    generator.load_state_dict(torch.load('/home/pzq/pose_warp/pretrain/PNG_s2e19.pkl'))

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
    src_imgs = pre.read_color(data_json_path, width, height, src_dir, start=20000, end=21000) / 255
    print('Read in src images done')
    pose_map = pre.read_gray(data_json_path, width, height, pose_map_dir, start=20000, end=21000) / 255
    tgt_masks = pre.read_gray(data_json_path, width, height, tgt_mask_dir, start=20000, end=21000) / 255
    print('Gray images reading done')
    warped_imgs = pre.read_morphed_img_and_group(warp_dir, data_json_path, match_path, width, height, num=2, start=20000, end=21000) / 255
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
            batch_in[instance_num * i + j, :3, :, :] = warped_imgs[index, j, :, :, :] * tgt_mask \
                                                       + (1 - tgt_mask) * tgt_img
            batch_mask[instance_num * i + j, :, :] = tgt_mask
            # batch_in[instance_num * i + j, :3, :, :] = warped_imgs[index, j, :, :, :]
    # print("End of prepare")

    # Start testing
    batch_in = torch.tensor(batch_in).float().cuda()
    with torch.no_grad():
        out = generator(batch_in)

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
    # run()
    out_model()

