import collections
import random
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import mmcv

interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class LoadImage(object):
    def __call__(self, results):
        img = results["img"]
        if isinstance(img, str):
            img = cv2.imread(img)
        assert isinstance(img, np.ndarray)
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


class Resize(object):
    def __init__(self, scale_list, keep_aspect=True, interpolation="bilinear"):
        if isinstance(scale_list, tuple):
            scale_list = [scale_list]
        assert isinstance(scale_list, list)
        self.scale_list = scale_list
        self.interpolation = interp_codes[interpolation]
        self.keep_aspect = keep_aspect

    def _random_scale(self, results):
        scale = random.choice(self.scale_list)
        results["scale"] = scale

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        img = results["img"]
        if self.keep_aspect:
            img, _ = mmcv.imrescale(img, results["scale"], return_scale=True)
            new_h, new_w = img.shape[:2]
            h, w = results["img"].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, results["scale"], return_scale=True
            )
        results["img"] = img

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results["img_shape"] = img.shape
        results["pad_shape"] = img.shape
        results["scale_factor"] = scale_factor
        results["keep_aspect"] = self.keep_aspect

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        if "bboxes" in results:
            img_shape = results["img_shape"]
            bboxes = results["bboxes"] * results["scale_factor"]
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results["bboxes"] = bboxes

    def __call__(self, results):
        if "scale" not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        return results


class Pad(object):
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        img = results["img"]
        if self.size is not None:
            padded_img = mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val
            )
        results["img"] = padded_img
        results["pad_shape"] = padded_img.shape
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        return results


class ToTensor(object):
    def _to_tensor(self, data):
        """Convert objects of various python types to :obj:`torch.Tensor`.
        Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
        :class:`Sequence`, :class:`int` and :class:`float`.
        Args:
            data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
                be converted.
        """

        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, Sequence) and not mmcv.is_str(data):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")

    def __call__(self, results):
        img = results["img"]
        if img.ndim < 3:
            img = np.expand_dims(img, -1)
        results["img"] = self._to_tensor(img.transpose(2, 0, 1))
        if "bboxes" in results:
            results["bboxes"] = self._to_tensor(results["bboxes"])
        if "labels" in results:
            results["labels"] = self._to_tensor(results["labels"])
        return results


class Normalize(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = mmcv.imnormalize(
            results["img"], self.mean, self.std, self.to_rgb
        )
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


class Collect(object):
    def __init__(
        self,
        keys=("img", "bboxes", "labels"),
        meta_keys=(
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ),
    ):
        assert isinstance(keys, tuple)
        assert isinstance(meta_keys, tuple)
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            img_metas[key] = results[key]
        data["img_metas"] = img_metas
        for key in self.keys:
            data[key] = results[key]
        return data


class Collate(object):
    def __call__(self, batch):
        img_list = []
        bboxes_list = []
        labels_list = []
        img_metas_list = []
        for b in batch:
            img_list.append(b["img"])
            bboxes_list.append(b["bboxes"])
            labels_list.append(b["labels"])
            img_metas_list.append(b["img_metas"])

        batch = len(img_list)
        sizes = [img.shape for img in img_list]
        channels, hegihts, widths = zip(*sizes)
        channel = channels[0]
        max_h = max(hegihts)
        max_w = max(widths)
        imgs = torch.zeros((batch, channel, max_h, max_w))
        for i, img in enumerate(img_list):
            c, h, w = img.shape
            imgs[i, :c, :h, :w] = img
        return imgs, bboxes_list, labels_list, img_metas_list


def build_transforms(cfg):
    return Compose(
        [
            Resize(cfg.train_size, cfg.keep_aspect),
            Normalize(cfg.mean, cfg.std, to_rgb=True),
            Pad(size_divisor=cfg.size_divisor),
            ToTensor(),
            Collect(),
        ]
    )


def build_test_transfrom(cfg):
    return Compose(
        [
            LoadImage(),
            Resize(cfg.test_size, cfg.keep_aspect),
            Normalize(cfg.mean, cfg.std, to_rgb=True,),
            Pad(size_divisor=cfg.size_divisor),
            ToTensor(),
            Collect(keys=("img",)),
        ]
    )
