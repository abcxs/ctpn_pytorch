import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import constant_init, kaiming_init
from mmcv.ops import bbox_overlaps
from mmcv.ops.nms import nms
from models.mobilenet import *
from models.resnet import *
from models.shufflenetv2 import *
from models.vgg import *
from utils.utils import multi_apply, unmap

from utils.accuracy import accuracy
from utils.anchor_generator import AnchorGenerator
from utils.bbox_coder import decode, encode
from utils.sampler import RandomSampler


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=True,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Im2col(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(Im2col, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        n, _, h, w = x.shape[-2:]
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x = x.reshape((n, -1, h, w))
        return x


class CTPN(nn.Module):
    def __init__(self, cfg):
        super(CTPN, self).__init__()
        base_model = cfg.base_model
        pretrained = cfg.pretrained
        self.num_classes = cfg.num_classes
        scales = [[cfg.width, h] for h in cfg.heights]
        self.anchor_generator = AnchorGenerator(cfg.stride, scales)
        self.cfg = cfg

        self.cnn = nn.Sequential()
        if "vgg" in base_model:
            self.cnn.add_module(
                base_model, globals().get(base_model)(pretrained=pretrained)
            )
        elif "mobile" in base_model:
            self.cnn.add_module(
                base_model, globals().get(base_model)(pretrained=pretrained)
            )
        elif "shuffle" in base_model:
            self.cnn.add_module(
                base_model, globals().get(base_model)(pretrained=pretrained)
            )
        elif "resnet" in base_model:
            self.cnn.add_module(
                base_model,
                globals().get(base_model)(pretrained=pretrained, model_name=base_model),
            )
        else:
            print("not support this base model")

        # self.rnn = nn.Sequential()
        # self.rnn.add_module("im2col", Im2col((3, 3), (1, 1), (1, 1)))

        self.rpn = BasicConv(512, 512, 3, 1, 1, bn=False)
        # self.brnn = nn.GRU(512, 128, bidirectional=True)
        self.brnn = nn.LSTM(512, 128, bidirectional=True)
        self.rnn_fc = BasicConv(256, 512, 1, 1, relu=True, bn=False)

        self.rpn_reg = nn.Conv2d(512, 4 * self.anchor_generator.num_base_anchors, 1)
        self.rpn_cls = nn.Conv2d(
            512, self.num_classes * self.anchor_generator.num_base_anchors, 1
        )

        self.init_weights()
        if cfg.freeze_norm:
            self.freeze_norm()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def freeze_norm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.cnn(x)
        x = self.rpn(x)

        b, c, h, w = x.size()
        x = x.permute(3, 0, 2, 1).reshape(w, b * h, c)
        x, _ = self.brnn(x)
        c = x.size()[-1]
        x = x.reshape(w, b, h, c).permute(1, 3, 2, 0).contiguous()

        x = self.rnn_fc(x)

        rpn_reg = self.rpn_reg(x)
        rpn_cls = self.rpn_cls(x)
        return rpn_cls, rpn_reg

    def get_anchors(self, featmap_size, img_metas, device):
        num_imgs = len(img_metas)
        anchors = self.anchor_generator.grid_anchors(featmap_size, device)
        anchor_list = [anchors for _ in range(num_imgs)]

        valid_flag_list = []
        for img_meta in img_metas:
            valid_flag_list.append(
                self.anchor_generator.valid_flags(
                    featmap_size, img_meta["pad_shape"], device
                )
            )
        return anchor_list, valid_flag_list

    def _anchor_inside_flags(
        self, flat_anchors, valid_flags, img_shape, allowed_border=0
    ):
        img_h, img_w = img_shape[:2]
        if allowed_border >= 0:
            inside_flags = (
                valid_flags
                & (flat_anchors[:, 0] >= -allowed_border)
                & (flat_anchors[:, 1] >= -allowed_border)
                & (flat_anchors[:, 2] < img_w + allowed_border)
                & (flat_anchors[:, 3] < img_h + allowed_border)
            )
        else:
            inside_flags = valid_flags
        return inside_flags

    def _get_target_single(
        self, flat_anchors, valid_flags, gt_bboxes, gt_labels, img_meta
    ):
        inside_flags = self._anchor_inside_flags(
            flat_anchors,
            valid_flags,
            img_meta["img_shape"][:2],
            self.cfg.allowed_border,
        )
        anchors = flat_anchors[inside_flags]

        assigned_gt_inds = flat_anchors.new_full(
            (anchors.shape[0],), -1, dtype=torch.long
        )

        # assign anchor 1, 0, -1
        # 1 for foreground
        # 0 for background
        # -1 for ignore
        overlaps = bbox_overlaps(gt_bboxes, anchors)
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        assigned_gt_inds[max_overlaps < self.cfg.neg_iou_thr] = 0
        pos_inds = max_overlaps > self.cfg.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        for i in range(len(gt_bboxes)):
            if gt_max_overlaps[i] > self.cfg.min_pos_iou:
                if self.cfg.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_max_overlaps[i]] = i + 1

        assigned_labels = assigned_gt_inds.new_full((flat_anchors.shape[0],), -1)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        # sample
        sampler = RandomSampler(
            self.cfg.rpn_batch, self.cfg.pos_fraction, self.cfg.neg_pos_ub
        )
        # idx in anchors
        pos_inds, neg_inds = sampler.sample(assigned_gt_inds)

        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((anchors.shape[0],), 0, dtype=torch.long)
        label_weights = anchors.new_zeros(anchors.shape[0], dtype=torch.float)

        pos_bbox_targets = encode(
            anchors[pos_inds], gt_bboxes[assigned_gt_inds[pos_inds] - 1]
        )
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = bbox_weights.new_tensor(self.cfg.rpn_bbox_weights)

        side_inds = torch.nonzero(
            assigned_gt_inds[pos_inds] == 1, as_tuple=False
        ).squeeze()
        side_inds = pos_inds[side_inds]
        if self.cfg.side_refine:
            bbox_weights[side_inds, 0] = self.cfg.side_weights

        labels[pos_inds] = 1
        label_weights[pos_inds] = self.cfg.pos_weights
        label_weights[neg_inds] = self.cfg.neg_weights

        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

    def get_targets(
        self, anchor_list, valid_flag_list, gt_bboxes_list, gt_labels_list, img_metas
    ):
        (
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
        )
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels = torch.stack(all_labels, 0)
        label_weights = torch.stack(all_label_weights, 0)
        bbox_targets = torch.stack(all_bbox_targets, 0)
        bbox_weights = torch.stack(all_bbox_weights, 0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_pos,
            num_total_neg,
        )

    def loss(self, cls_score, bbox_pred, gt_bboxes, gt_labels, img_metas):
        featmap_size = cls_score.size()[-2:]
        device = cls_score.device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_size, img_metas, device=device
        )
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_pos,
            num_total_neg,
        ) = self.get_targets(
            anchor_list, valid_flag_list, gt_bboxes, gt_labels, img_metas
        )

        num_total_samples = num_total_pos + num_total_neg

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        acc = accuracy(cls_score, labels)
        loss_cls = (
            F.cross_entropy(cls_score, labels, reduction="none")
            * label_weights
            / num_total_samples
        )

        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = (
            F.smooth_l1_loss(bbox_pred, bbox_targets, reduction="none")
            * bbox_weights
            / num_total_pos
        )
        return (
            loss_cls.sum() * self.cfg.cls_weight,
            loss_bbox.sum() * self.cfg.reg_weight,
            acc,
        )

    def simple_test(
        self, x, img_metas,
    ):
        cls_score, bbox_pred = self.forward(x)
        device = cls_score.device

        featmap_size = cls_score.size()[-2:]
        anchors = self.anchor_generator.grid_anchors(featmap_size, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            pad_shape = img_metas[img_id]["pad_shape"]
            valid_flag = self.anchor_generator.valid_flags(
                featmap_size, pad_shape, device=device
            )
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            dets = self._get_bboxes(
                cls_score[img_id],
                bbox_pred[img_id],
                anchors,
                valid_flag,
                img_shape,
                scale_factor,
            )
            result_list.append(dets)
        return result_list

    def _get_bboxes(
        self, cls_score, bbox_pred, anchors, valid_flag, img_shape, scale_factor
    ):
        # filter outside anchors which are not trained
        anchors = anchors[valid_flag]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)
        cls_score = cls_score[valid_flag]
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        bbox_pred = bbox_pred[valid_flag]
        scores = cls_score.softmax(-1)

        id_ = scores[:, 1] > self.cfg.score_thre
        anchors = anchors[id_]
        bbox_pred = bbox_pred[id_]
        scores = scores[id_]

        bboxes = decode(anchors, bbox_pred, max_shape=img_shape)
        bboxes /= bboxes.new_tensor(scale_factor)

        dets, _ = nms(bboxes, scores[:, 1].contiguous(), self.cfg.nms_thre)
        sorted_indices = torch.argsort(dets[:, -1], descending=True)
        dets = dets[sorted_indices]
        return dets
