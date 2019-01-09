"""
This is the file for discriminator
Structure guided by DCGAN
What to modify:
    file layer may be better with convolution
    layer number
    channel number
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def discriminator_block(in_filters, out_filters, bn = True):
    """
    Basic block in discriminator
    """
    block = nn.Sequential(
        nn.Conv2d(in_filters, out_filters, 3, 2, 1),
        nn.LeakyReLU(0.2, inplace=True)
    )
    if bn:
        block = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_filters, 0.8),
        )
    return block


class Discriminator(nn.Module):
    def __init__(self, img_height, img_width, useBN=True):
        super(Discriminator, self).__init__()
        self.block1 = discriminator_block(3, 32, useBN)
        self.block2 = discriminator_block(32, 64, useBN)
        self.block3 = discriminator_block(64, 128, useBN)
        self.block4 = discriminator_block(128, 256, useBN)
        self.block5 = discriminator_block(256, 512, useBN)

        ds_height = img_height // 2**5
        ds_width = img_width // 2**5
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_height * ds_width, 1),
                                       nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                    nn.init.xavier_uniform(m.weight)


    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
