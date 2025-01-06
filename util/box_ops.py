# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import math

import numpy as np
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def ciou(boxes1, boxes2):
    """
    Complete Intersection over Union (CIoU) Loss.
    The boxes should be in [x0, y0, x1, y1] format.

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2).
    """
    # 确保所有盒子的宽度和高度是有效的
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # 计算两个盒子集合之间的IoU
    iou, union = box_iou(boxes1, boxes2)

    # 计算所有盒子对的交集的左上角坐标和右下角坐标
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    # 计算交集区域的宽度和高度，结果为[N,M,2]
    wh = (rb - lt).clamp(min=0)
    # 计算交集区域的面积
    area = wh[:, :, 0] * wh[:, :, 1]

    # 计算每个预测框和真实框的中心点坐标
    center_x1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
    center_y1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    center_x2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
    center_y2 = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # 计算中心点距离
    center_distance = torch.sqrt((center_x1[:, None] - center_x2[None, :]) ** 2 +
                                 (center_y1[:, None] - center_y2[None, :]) ** 2)

    # 计算对角线长度
    diagonal_length = torch.sqrt((boxes1[:, 2] - boxes1[:, 0]) ** 2 +
                                 (boxes1[:, 3] - boxes1[:, 1]) ** 2)[:, None] + \
                      torch.sqrt((boxes2[:, 2] - boxes2[:, 0]) ** 2 +
                                 (boxes2[:, 3] - boxes2[:, 1]) ** 2)[None, :]

    # 计算宽高比损失
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]
    # 确保不除以零
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(w2[None, :] / torch.clamp(h2[None, :], min=1e-6)) -
        torch.atan(w1[:, None] / torch.clamp(h1[:, None], min=1e-6)), 2)

    # 计算CIoU
    ciou = iou - (center_distance / diagonal_length) - v

    return ciou



def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # 确保所有盒子的宽度和高度是有效的，即右下角坐标大于等于左上角坐标
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # 计算两个盒子集合之间的IOU（Intersection over Union，即Jaccard指数）
    iou, union = box_iou(boxes1, boxes2)

    # 计算所有盒子对的交集的左上角坐标
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # 计算所有盒子对的并集的右下角坐标
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 计算交集区域的宽度和高度，结果为[N,M,2]
    wh = (rb - lt).clamp(min=0)
    # 计算交集区域的面积
    area = wh[:, :, 0] * wh[:, :, 1]



    giou = iou - (area - union) / area



    # 返回修正后的IOU值，这里使用了面积差与交集面积的比值进行调整
    return giou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
