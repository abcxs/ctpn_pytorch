import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transforms, side_refine=False):
        self.img_dir = os.path.join(root, "imgs")
        self.label_dir = os.path.join(root, "labels")
        self.img_files = [
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ]
        self.transforms = transforms
        self.side_refine = side_refine
        self._set_group_flag()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_file = self.img_files[i]
            img = Image.open(img_file)
            w, h = img.size
            if w / h > 1:
                self.flag[i] = 1

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img = cv2.imread(img_file)

        base_name = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(self.label_dir, "%s.txt" % base_name)
        with open(label_file) as f:
            bboxes = f.read().split("\n")
        bboxes = [box.strip().split(" ") for box in bboxes]
        bboxes = [list(map(float, box)) for box in bboxes if len(box) >= 4]
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[-1] == 4 or not self.side_refine:
            labels = np.zeros(bboxes.shape[0], dtype=np.int64)
        else:
            labels = bboxes[:, -1].astype(np.int64)
        bboxes = bboxes[:, :4]
        results = {
            "img": img,
            "bboxes": bboxes,
            "labels": labels,
            "ori_shape": img.shape,
        }
        results = self.transforms(results)
        return results
