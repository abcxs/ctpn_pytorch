import xml.etree.ElementTree as ET
import os
import cv2
from tqdm import tqdm
input_dir = '/home/zhoufeipeng/data/VOC2007/Annotations'
img_dir = '/home/zhoufeipeng/data/VOC2007/JPEGImages'
output_dir = os.path.join(os.path.dirname(input_dir), 'labels')
visual_dir = os.path.join(os.path.dirname(input_dir), 'visual')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(visual_dir, exist_ok=True)
input_files = os.listdir(input_dir)
input_files = [os.path.join(input_dir, f) for f in input_files]
for input_file in tqdm(input_files):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    tree = ET.parse(input_file)
    root = tree.getroot()

    img_path = os.path.join(img_dir, '%s.jpg' % base_name)
    img = cv2.imread(img_path)

    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bnd_box = obj.find('bndbox')
        xmin = bnd_box.find('xmin').text
        ymin = bnd_box.find('ymin').text
        xmax = bnd_box.find('xmax').text
        ymax = bnd_box.find('ymax').text
        bbox = '%s %s %s %s' % (xmin, ymin, xmax, ymax)
        bboxes.append(bbox)

        x1 = int(float(xmin))
        y1 = int(float(ymin))
        x2 = int(float(xmax))
        y2 = int(float(ymax))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    with open(os.path.join(output_dir, '%s.txt' % base_name), 'w') as f:
        f.write('\n'.join(bboxes))
    cv2.imwrite(os.path.join(visual_dir, '%s.png' % base_name), img)
    