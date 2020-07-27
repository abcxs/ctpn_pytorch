import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
from prepare.utils import orderConvex, shrink_poly

# input_dir = "/home/zhoufeipeng/data/pdf_tmp"
# output_dir = "/home/zhoufeipeng/data/pdf_split1"
# min_size = 1200
# max_size = 2000
# symbol = ' '
input_dir = '/home/zhoufeipeng/data/mlt'
output_dir = '/home/zhoufeipeng/data/mlt_split'
min_size = 600
max_size = 1200
symbol = ','
min_poly_size = 5

img_files = os.listdir(os.path.join(input_dir, "imgs"))
img_files.sort()

if not os.path.exists(os.path.join(output_dir, "imgs")):
    os.makedirs(os.path.join(output_dir, "imgs"))
if not os.path.exists(os.path.join(output_dir, "labels")):
    os.makedirs(os.path.join(output_dir, "labels"))
if not os.path.exists(os.path.join(output_dir, "visual")):
    os.makedirs(os.path.join(output_dir, "visual"))
# img_files = ["12_10.png"]
f = False
for img_file in tqdm(img_files):
    bfn, ext = os.path.splitext(img_file)
    if ext.lower() not in [".jpg", ".png"]:
        continue

    label_path = os.path.join(input_dir, "labels", "%s.txt" % bfn)
    img_path = os.path.join(input_dir, "imgs", img_file)
    if not (os.path.exists(label_path) and os.path.exists(img_path)):
        continue

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    im_sclae = min(max_size / max(h, w), min_size / min(h, w))
    new_h = int(h * im_sclae)
    new_w = int(w * im_sclae)

    re_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    re_h, re_w, _ = re_img.shape
    visual_img = re_img.copy()

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    polys = []

    with open(label_path) as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.strip().split(symbol)
        x1, y1, x3, y3 = map(float, split_line)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x3 = min(x3, gray_img.shape[1])
        y3 = min(y3, gray_img.shape[0])
        assert x3 >= x1 and y3 >= y1

        x1_ori, y1_ori, x3_ori, y3_ori = int(x1), int(y1), int(x3), int(y3)
        if x3_ori - x1_ori < min_poly_size or y3_ori - y1_ori < min_poly_size:
            continue
        
        temp = gray_img[y1_ori:y3_ori, x1_ori:x3_ori]
        _, temp_img = cv2.threshold(temp, 120, 255, cv2.THRESH_OTSU)
        temp_img = np.invert(temp_img)
        temp_img = cv2.morphologyEx(
            temp_img, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)
        )
        non_zero_area = cv2.findNonZero(temp_img)
        x_, y_, w_, h_ = cv2.boundingRect(non_zero_area)

        if w_ < min_poly_size or h_ < min_poly_size:
            continue
        x1_ = max(x1 + x_ - 16, 0)
        y1_ = max(y1 + y_ - 2, 0)
        x3_ = min(x1 + x_ + w_ + 4, w)
        y3_ = min(y1 + y_ + h_ + 2, h)

        x1, y1, x3, y3 = x1_, y1_, x3_, y3_

        x2 = x3
        y2 = y1
        x4 = x1
        y4 = y3
        poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape(4, 2)
        poly[:, 0] = poly[:, 0] / w * re_w
        poly[:, 1] = poly[:, 1] / h * re_h
        poly = orderConvex(poly)

        cv2.polylines(
            visual_img,
            [poly.astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=(0, 255, 0),
            thickness=2,
        )

        polys.append(poly)

    res_polys = []
    res_labels = []
    for poly in polys:

        res, labels = shrink_poly(poly)

        # for p in res:
        #     cv2.polylines(
        #         re_img,
        #         [p.astype(np.int32).reshape((-1, 1, 2))],
        #         True,
        #         color=(255, 0, 0),
        #         thickness=1,
        #     )

        res = res.reshape([-1, 4, 2])
        for r in res:
            x_min = np.min(r[:, 0])
            y_min = np.min(r[:, 1])
            x_max = np.max(r[:, 0])
            y_max = np.max(r[:, 1])

            res_polys.append([x_min, y_min, x_max, y_max])
        res_labels.extend(labels.tolist())

    cv2.imwrite(os.path.join(output_dir, "imgs", img_file), re_img)

    with open(os.path.join(output_dir, "labels", "%s.txt" % bfn), "w") as f:
        for p, l in zip(res_polys, res_labels):
            line = " ".join(list(map(str, p)))
            line += " %d" % l
            f.write(line + "\n")
        for p, l in zip(res_polys, res_labels):
            color = (255, 0, 0)
            if l == 1:
                color = (0, 0, 255)
            cv2.rectangle(
                visual_img, (p[0], p[1]), (p[2], p[3]), color=color, thickness=1,
            )
    cv2.imwrite(os.path.join(output_dir, "visual", img_file), visual_img)

