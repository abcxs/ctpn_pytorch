import numpy as np
import torch
from torch.nn.modules.utils import _pair


class AnchorGenerator(object):
    def __init__(self, stride, scales):
        self.stride = stride
        self.scales = torch.tensor(scales).reshape(-1, 2)
        self.base_anchors = self.gen_base_anchor()

    @property
    def num_base_anchors(self):
        return len(self.scales)

    def gen_base_anchor(self):
        x_ctr, y_ctr = (self.stride[0] - 1) * 0.5, (self.stride[1] - 1) * 0.5
        x1 = x_ctr - self.scales[:, 0] * 0.5
        y1 = y_ctr - self.scales[:, 1] * 0.5
        x2 = x_ctr + self.scales[:, 0] * 0.5
        y2 = y_ctr + self.scales[:, 1] * 0.5
        base_anchors = torch.stack((x1, y1, x2, y2), dim=-1)
        return base_anchors

    def grid_anchors(self, featmap_size, device="cuda"):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * self.stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * self.stride[1]
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_y = shift_y.reshape(-1)
        shift_x = shift_x.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1)

        base_anchors = self.base_anchors.to(device)
        shifts = shifts.type_as(base_anchors)
        # M, 4 + N, 4
        anchors = base_anchors[None, :, :] + shifts[:, None, :]
        anchors = anchors.view(-1, 4)
        return anchors

    def valid_flags(self, featmap_size, pad_shape, device="cuda"):
        feat_h, feat_w = featmap_size
        h, w = pad_shape[:2]
        valid_feat_h = min(int(np.ceil(h / self.stride[0])), feat_h)
        valid_feat_w = min(int(np.ceil(w / self.stride[1])), feat_w)
        assert valid_feat_h <= feat_h and valid_feat_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_feat_w] = 1
        valid_y[:valid_feat_h] = 1
        valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
        valid_y = valid_y.reshape(-1)
        valid_x = valid_x.reshape(-1)
        valid = valid_x & valid_y
        valid = (
            valid[:, None]
            .expand(valid.size(0), self.num_base_anchors)
            .contiguous()
            .view(-1)
        )
        return valid
