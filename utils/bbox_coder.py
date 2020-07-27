import torch
import numpy as np


def encode(bboxes, gt_bboxes):
    assert bboxes.size(0) == gt_bboxes.size(0)
    bboxes = bboxes.float()
    gt_bboxes = gt_bboxes.float()

    px = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    py = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    pw = bboxes[..., 2] - bboxes[..., 0]
    ph = bboxes[..., 3] - bboxes[..., 1]

    gx = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
    gy = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
    gw = gt_bboxes[..., 2] - gt_bboxes[..., 0]
    gh = gt_bboxes[..., 3] - gt_bboxes[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    return deltas


def decode(bboxes, deltas, max_shape=None, wh_ratio_clip=16 / 1000):
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    # dw[:, :] = 0
    # dx[:, :] = 0

    max_ratio = np.abs(np.log(wh_ratio_clip))

    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    # Compute center of each roi
    px = ((bboxes[:, 0] + bboxes[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((bboxes[:, 1] + bboxes[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (bboxes[:, 2] - bboxes[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (bboxes[:, 3] - bboxes[:, 1]).unsqueeze(1).expand_as(dh)

    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy

    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes
