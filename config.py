import os

# train
base_model = "resnet50"
pretrained = True
checkpoint = None
side_refine = True
freeze_norm = True
num_classes = 2

# optimizer
optimizer = "SGD"
lr = 0.001
momentum = 0.9
step_size = [8, 11]
max_epoch = 12
gamma = 0.1
weight_decay = 5e-4

# setting
temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
os.makedirs(temp_dir, exist_ok=True)
seed = 1
work_dir = None
save_interval = 1
iteration_show = 20
device = "cuda"

# dataset
train_root = "/home/zhoufeipeng/data/pdf_split1"
train_size = [(2000, 1200)]
test_size = (3000, 1400)
keep_aspect = True
size_divisor = 16
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]
to_rgb = True
batch_size = 2
num_workers = 2

# Anchor
heights = [8, 11, 16, 23, 33, 48, 68, 97, 139, 198]
width = 16
stride = [16, 16]
allowed_border = 0
neg_iou_thr = 0.5
pos_iou_thr = 0.7
min_pos_iou = 0.5
neg_pos_ub = -1
gt_max_assign_all = True
pos_fraction = 0.5
rpn_batch = 256
rpn_bbox_weights = (0, 1.0, 0, 1.0)
pos_weights = 1.0
neg_weights = 1.0
side_weights = 2.0
cls_weight = 1.0
reg_weight = 1.0


# test
nms_thre = 0.2
score_thre = 0.5

# cover
rpn_bbox_weights = (1.0, 1.0, 1.0, 1.0)
side_weights = 1.0