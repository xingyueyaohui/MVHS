"""
This is the file for computing loss
"""
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


class ConfidenceLoss(nn.Module):
    """
    This is the final module for confidence loss function
    """
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def forward(self, synth, target, predict):
        [batch_size, _, height, width] = predict.shape
        confidence = torch.abs(predict).view((batch_size, height, width))
        difference = synth - target
        difference = torch.sum(difference, dim=1)
        difference_map = torch.abs(difference)
        map = difference_map * confidence

        confidence_sum = torch.sum(confidence.view((batch_size, -1)), dim=1)
        map_sum = torch.sum(map.view((batch_size, -1)), dim=1)
        self.loss = torch.sum(map_sum / confidence_sum)
        return self.loss



