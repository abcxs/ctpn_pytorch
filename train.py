import argparse
import collections
import os
import time

import torch
from torch import optim
from torch.utils.data import DataLoader

import config as cfg
from datasets import Collate, GroupSampler, ImageDataset, build_transforms
from models.ctpn import CTPN
from utils.log_buffer import LogBuffer
from utils.logger_helper import get_root_logger
from utils.utils import (load_checkpoint, save_checkpoint, set_random_seed,
                         time2hms)


def main():
    args = get_args()

    assert not (args.resume_from and args.load_from)

    # work_dir
    work_dir = cfg.work_dir
    if args.work_dir:
        work_dir = args.work_dir
    if work_dir is None:
        work_dir = "./work_dir"
    os.makedirs(work_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    set_random_seed(cfg.seed, deterministic=args.deterministic)

    log_file = os.path.join(work_dir, "out.log")
    logger = get_root_logger(log_file=log_file)

    logger.info("work_dir: %s, log_file: %s" % (work_dir, log_file))

    transforms = build_transforms(cfg)
    dataset = ImageDataset(
        cfg.train_root, transforms=transforms, side_refine=cfg.side_refine
    )
    sampler = GroupSampler(dataset, cfg.batch_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, num_workers=cfg.num_workers, collate_fn=Collate(),
    )
    device = "cuda"
    device = torch.device(device)
    model = CTPN(cfg).to(device)

    only_weights = True
    checkpoint = cfg.checkpoint
    if args.load_from:
        checkpoint = args.load_from
    if args.resume_from:
        checkpoint = args.resume_from
        only_weights = False
    checkpoint = load_checkpoint(model, checkpoint, only_weights)

    if cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        # check_epoch
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.step_size, gamma=cfg.gamma
        )
    elif cfg.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = 0
    iteration = 0
    if "meta" in checkpoint:
        epoch = checkpoint["meta"]["epoch"]
        iteration = checkpoint["meta"]["iteration"]

    max_iterations = cfg.max_epoch * len(dataloader)
    log_buffer = LogBuffer()

    model.train()
    while epoch < cfg.max_epoch:
        data_start = time.time()
        for i, (imgs, gt_bboxes, gt_labels, img_metas) in enumerate(dataloader):
            data_end = time.time()
            data_time = data_end - data_start

            optimizer.zero_grad()
            imgs = imgs.to(device)
            gt_bboxes = [bboxes.to(device) for bboxes in gt_bboxes]
            gt_labels = [labels.to(device) for labels in gt_labels]

            cuda_start = time.time()
            rpn_cls, rpn_reg = model(imgs)

            cls_loss, reg_loss, acc = model.loss(
                rpn_cls, rpn_reg, gt_bboxes, gt_labels, img_metas
            )

            loss = cls_loss + reg_loss
            loss_ = {
                "cls_loss": cls_loss.item(),
                "reg_loss": reg_loss.item(),
                "loss": loss.item(),
            }
            log_buffer.update(loss_)

            loss.backward()
            optimizer.step()

            iteration += 1

            cuda_end = time.time()
            cuda_time = cuda_end - cuda_start

            time_ = {"data_time": data_time, "cuda_time": cuda_time}
            log_buffer.update(time_)

            if isinstance(acc, collections.abc.Sequence):
                acc = acc[0]

            acc_ = {"accuracy": acc}
            log_buffer.update(acc_)

            if iteration % cfg.iteration_show == 0:
                log_buffer.average()
                eta = (max_iterations - iteration) * (
                    log_buffer.output["data_time"] + log_buffer.output["cuda_time"]
                )
                h, m, s = time2hms(eta)
                eta = "%d h %d m %d s" % (h, m, s)

                info = ""
                for k, v in log_buffer.output.items():
                    if "loss" in k or "accuracy" in k:
                        info += "%s: %.4f " % (k, v)

                log = f"[{epoch + 1}/{cfg.max_epoch}][{i + 1}/{len(dataloader)}] iteration: {iteration} data_time: {data_time:.2} cuda_time: {cuda_time:.2} eta: {eta} {info}"
                logger.info(log)
                log_buffer.clear()

            data_start = time.time()
        epoch += 1
        if cfg.optimizer == "SGD":
            scheduler.step()

        if epoch % cfg.save_interval == 0 or epoch == cfg.max_epoch:
            meta = {"epoch": epoch, "iteration": iteration}
            if cfg.optimizer == "Adam":
                scheduler = None
            save_checkpoint(
                os.path.join(work_dir, "epoch_%d.pth" % epoch),
                model,
                optimizer,
                scheduler,
                meta,
            )


def get_args():
    parser = argparse.ArgumentParser("ctpn")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument("--load-from", help="the checkpoint file to load from")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
