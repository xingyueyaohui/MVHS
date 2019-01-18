from model.PNNet import Res_Generator as Generator
from model.DC_descriminator import Discriminator as Discriminator
from lib.dataset import Dataset, Person
import lib.move_util as move
import lib.data_generator as dg
import lib.prework as pre
from model.metrics import ssim as ssim

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

def print_out():
    src_dir = '/home/dongkai/pzq/data/market-morph/'
    dst_dir = '/home/dongkai/pzq/data/market-refine/'
    pose_dir = '/home/dongkai/pzq/data/market-pose-map/'
    tgt_mask_dir = '/home/dongkai/pzq/data/market-mask/'
    tgt_dir = '/home/dongkai/pzq/data/market-gt/'
    image_list = os.listdir(src_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    network = Generator(64, 6).cuda()
    network.load_state_dict(torch.load('/home/dongkai/pzq/MVHS/base_model/G.pkl'))
    for param in network.parameters():
        param.requires_grad = False

    for i, image_name in enumerate(image_list):
        input = np.zeros((1, 4, 128, 64))
        img = np.transpose(cv2.imread(src_dir + image_name) / 255, (2, 0, 1))
        img_name = image_name[:-5] + '.jpg'
        pose = cv2.imread(pose_dir + img_name, 0) / 255
        mask = cv2.imread(tgt_mask_dir + img_name, 0) / 255
        tgt_img = np.transpose(cv2.imread(tgt_dir + img_name) / 255, (2, 0, 1))
        img = img * mask + tgt_img * (1 - mask)

        input[0, :3, :, :] = img
        input[0, 3, :, :] = pose
        input = torch.tensor(input).float().cuda()
        out = network(input)
        out = (np.array(out.detach().cpu()) * 255).astype(np.uint8)
        out = np.transpose(out[0], (1, 2, 0))
        cv2.imwrite(dst_dir + image_name, out)

        if i % 200 == 0:
            print(i)

def compute_ave_ssim():
    src_dir = '/home/dongkai/pzq/data/market-refine/'
    tgt_dir = '/home/dongkai/pzq/data/market-gt/'
    image_list = os.listdir(src_dir)
    image_list = sorted(image_list, key=lambda x:x)
    total_ssim = 0
    num = 2000
    test_list = np.arange(40000, 40000 + num)
    np.random.shuffle(test_list)

    img0 = np.zeros((num, 3, 128, 64))
    img1 = np.zeros((num, 3, 128, 64))
    for i in range(num):
        image_name = image_list[test_list[i]]
        img0[i] = np.transpose(cv2.imread(src_dir + image_name) / 255, (2, 0, 1))
        img1[i] = np.transpose(cv2.imread(tgt_dir + image_name[:-5] + '.jpg') / 255, (2, 0, 1))
    
    img0 = torch.tensor(img0).cuda()
    img1 = torch.tensor(img1).cuda()

    print('Start')

    for j in range(int(num / 2000)):
        total_ssim += ssim(img1[j*2000:(j+1)*2000], img0[j*2000:(j+1)*2000])

    print(total_ssim / (num / 2000))


def compute_max_ssim():
    src_dir = '/home/dongkai/pzq/data/market-refine/'
    tgt_dir = '/home/dongkai/pzq/data/market-gt/'
    image_list = os.listdir(src_dir)
    image_list = sorted(image_list, key=lambda x:x)
    total_ssim = 0
    num = 1000
    test_list = np.arange(20000, 20000 + num)
    np.random.shuffle(test_list)

    
    for i in range(num):
        img0 = np.zeros((2, 3, 128, 64))
        img1 = np.zeros((1, 3, 128, 64))
        image_name = image_list[test_list[i]][:-5]
        img1[0] = np.transpose(cv2.imread(tgt_dir + image_name + '.jpg') / 255, (2, 0, 1))
        for j in range(2):
            img_name = image_name + str(j) + '.jpg'
            img0[j, :, :, :] = np.transpose(cv2.imread(src_dir + img_name) / 255, (2, 0, 1))

        img1 = torch.tensor(img1).cuda().float()
        img0 = torch.tensor(img0).cuda().float()
        
        ssim0 = ssim(img1, img0[0:1, :, :, :])
        ssim1 = ssim(img1, img0[1:, :, :, :])
        total_ssim += torch.max(ssim0, ssim1)

    total_ssim = total_ssim / num

    print(total_ssim)



if __name__ == '__main__':
    # print_out()
    compute_max_ssim()    
    