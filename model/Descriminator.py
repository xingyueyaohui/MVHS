"""
This is the file for discriminator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def discriminator_block(in_filters, out_filters, bn = True):
    """
    Basic block in discriminator
    """
    block = [
        nn.Conv2d(in_filters, out_filters, 3, 2, 1),
        nn.LeakyReLU(0.2, inplace=True)
    ]
    if bn:
        block.append(nn.BatchNorm2d(out_filters, 0.8))
    return block


class Discriminator(nn.Module):
    def __init__(self, img_height, img_width, useBN=True):
        super(Discriminator, self).__init__()
        self.block1 = discriminator_block(3, 16, useBN)
        self.block2 = discriminator_block(16, 32, useBN)
        self.block3 = discriminator_block(32, 64, useBN)
        self.block4 = discriminator_block(64, 128, useBN)

        ds_height = img_height // 2**4
        ds_width = img_width // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_height * ds_width, 1),
                                       nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    nn.init.xavier_uniform(m.weight)


    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
