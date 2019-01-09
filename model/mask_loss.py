import torch
import torch.nn as nn

class MaskLoss(nn.Module):
    def __init__(self, criterion):
        super(MaskLoss, self).__init__()
        self.criterion = criterion

    def forward(self, input, target, mask_src, mask_tgt):
        input = input.transpose(0, 1)
        # mask_src = mask_src.transpose(0, 1)
        # mask_tgt = mask_tgt.transpose(0, 1)
        target = target.transpose(0, 1)
        src_masked = input * mask_src
        tgt_masked = target * mask_tgt
        src_masked = src_masked.transpose(0, 1)
        tgt_masked = tgt_masked.transpose(0, 1)
        self.loss = self.criterion(src_masked, tgt_masked)
        return self.loss