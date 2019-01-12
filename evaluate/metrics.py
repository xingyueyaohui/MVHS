"""
This file is for evaluating the performance
Including the metrics of:
    SSIM
"""
from evaluate.ssim import ssim as ssim
import math


def compute_avg_ssim(src_batch, tgt_batch):
    """
    compute the average ssim
    """
    batch_size = src_batch.shape[0]
    sum_ssim = 0.0
    for i in range(batch_size):
        sum_ssim += ssim(src_batch[i], tgt_batch[i])
    return sum_ssim / batch_size


def oracle_ssim(src_batch, tgt_batch, num=2):
    """
    compute the best ssim from every group so as to show the effectiveness of our approach
    """
    batch_size = src_batch.shape[0]
    sum_ssim = 0.0
    image_num = int(batch_size / num)
    for j in range(image_num):
        i = num * j
        sum_ssim += max(ssim(src_batch[i], tgt_batch[i]), ssim(src_batch[i + 1], tgt_batch[i + 1]))
    return sum_ssim / batch_size
