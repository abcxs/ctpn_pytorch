import argparse
import os

import cv2
import numpy as np
import torch
import pdf2image
from PyPDF2 import PdfFileReader, PdfFileWriter

import config as cfg
from datasets import build_test_transfrom
from models.ctpn import CTPN
from utils.logger_helper import get_root_logger
from utils.text_connector.detectors import TextDetector
from utils.utils import (
    generate_request_id,
    load_checkpoint,
    time_record,
    enable_time_record,
)


def init(cfg, device, checkpoint):
    model = CTPN(cfg)
    load_checkpoint(model, checkpoint, only_weights=True, map_location=device)
    model.to(device)
    model.eval()
    return model


@enable_time_record
def process_input(filepath, password=""):
    ext = os.path.splitext(filepath)[-1]
    if ext in [".png", ".jpg", ".JPEG"]:
        return [filepath]
    if ext in [".pdf"]:
        out_paths = []
        infile = PdfFileReader(open(filepath, "rb"), strict=False)
        if infile.isEncrypted:
            infile.decrypt(password)
        page_nums = infile.getNumPages()
        temp_dir = os.path.join(cfg.temp_dir, generate_request_id())
        os.makedirs(temp_dir, exist_ok=True)
        for i in range(page_nums):
            pdf_page_path = os.path.join(temp_dir, "page-%d.pdf" % i)
            image_path = os.path.join(temp_dir, "page-%d.png" % i)

            p = infile.getPage(i)
            outfile = PdfFileWriter()
            outfile.addPage(p)
            with open(pdf_page_path, "wb") as f:
                outfile.write(f)

            pages = pdf2image.convert_from_path(pdf_page_path, dpi=400)
            for page in pages:
                page.save(image_path)
            out_paths.append(image_path)
        return out_paths
    return []


def inference_simple(
    model, input_dir, output_dir=None, show=False, device=None, logger=None
):
    if not logger:
        logger = get_root_logger()
    if os.path.isfile(input_dir):
        input_files = [input_dir]
    else:
        input_files = os.listdir(input_dir)
        input_files = [os.path.join(input_dir, f) for f in input_files]

    transforms = build_test_transfrom(cfg)

    id_ = 0
    for input_file in input_files:
        logger.info("begin process %s" % input_file)
        imgs_path = process_input(input_file)
        for img_path in imgs_path:
            logger.info("begin infer %s" % img_path)
            with torch.no_grad():
                with time_record("transform"):
                    src = {"img": img_path}
                    dst = transforms(src)
                    img = dst["img"][None].to(device)
                    img_metas = [dst["img_metas"]]
                with time_record("inference sample"):
                    results = model.simple_test(img, img_metas)
                if show and output_dir:
                    output_path = os.path.join(output_dir, "%d.png" % id_)
                    show_result(img_path, results[0], output_path)
                    id_ += 1


def show_result(img_path, results, output_path):
    img = cv2.imread(img_path)
    results = results.cpu().numpy()
    bboxes = results[:, :-1].astype(np.int32)
    # for bbox in bboxes:
    #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))

    boxes = post_process(results, img.shape[:2])
    for i, box in enumerate(boxes):
        cv2.polylines(
            img,
            [box[:8].astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=(0, 255, 0),
            thickness=2,
        )

    cv2.imwrite(output_path, img)


def post_process(results, size):
    bboxes = results[:, :-1]
    scores = results[:, -1]
    text_detector = TextDetector()
    txt_rect, _ = text_detector.detect(bboxes, scores, size)
    return txt_rect


def get_args():
    parser = argparse.ArgumentParser("ctpn")
    parser.add_argument("input_dir", help="the dir to process")
    parser.add_argument("--output_dir", help="ouput dir to save", default=None)
    parser.add_argument(
        "--show", action="store_true", help="visual result", default=False
    )
    parser.add_argument("--checkpoint", help="the checkpoint file to load")
    parser.add_argument(
        "--device", help="inference with device, cpu or gpu", default="cpu"
    )

    return parser.parse_args()


def main():
    args = get_args()
    input_dir = args.input_dir

    device = cfg.device
    if args.device:
        device = args.device
    deivce = torch.device(device)

    checkpoint = cfg.checkpoint
    if args.checkpoint:
        checkpoint = args.checkpoint

    output_dir = None
    if args.output_dir:
        output_dir = args.output_dir
    if output_dir is None:
        output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)

    logger = get_root_logger()

    logger.info("init detector, device: %s, checkpoint: %s" % (device, checkpoint))
    model = init(cfg, device, checkpoint)

    logger.info("inference %s" % input_dir)
    inference_simple(
        model, input_dir, output_dir, args.show, device=device, logger=logger
    )


if __name__ == "__main__":
    main()
