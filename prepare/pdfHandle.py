import os
from tempfile import TemporaryDirectory
import cv2
import pdf2image
import numpy as np
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTImage,
    LTTextLineHorizontal,
    LTTextLineVertical,
)
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser
from PyPDF2 import PdfFileReader, PdfFileWriter

min_num = 5


def get_page_layout(
    filename,
    char_margin=1.0,
    line_margin=0.5,
    word_margin=0.1,
    detect_vertical=True,
    all_texts=True,
):
    """Returns a PDFMiner LTPage object and page dimension of a single
    page pdf. See https://euske.github.io/pdfminer/ to get definitions
    of kwargs.
    Parameters
    ----------
    filename : string
        Path to pdf file.
    char_margin : float
    line_margin : float
    word_margin : float
    detect_vertical : bool
    all_texts : bool
    Returns
    -------
    layout : object
        PDFMiner LTPage object.
    dim : tuple
        Dimension of pdf page in the form (width, height).
    """
    with open(filename, "rb") as f:
        parser = PDFParser(f)
        document = PDFDocument(parser)
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed
        laparams = LAParams(
            char_margin=char_margin,
            line_margin=line_margin,
            word_margin=word_margin,
            detect_vertical=detect_vertical,
            all_texts=all_texts,
        )
        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            width = layout.bbox[2]
            height = layout.bbox[3]
            dim = (width, height)
        return layout, dim


def get_text_objects(layout, ltype="char", t=None):
    """Recursively parses pdf layout to get a list of
    PDFMiner text objects.
    Parameters
    ----------
    layout : object
        PDFMiner LTPage object.
    ltype : string
        Specify 'char', 'lh', 'lv' to get LTChar, LTTextLineHorizontal,
        and LTTextLineVertical objects respectively.
    t : list
    Returns
    -------
    t : list
        List of PDFMiner text objects.
    """
    if ltype == "char":
        LTObject = LTChar
    elif ltype == "image":
        LTObject = LTImage
    elif ltype == "horizontal_text":
        LTObject = LTTextLineHorizontal
    elif ltype == "vertical_text":
        LTObject = LTTextLineVertical
    if t is None:
        t = []
    try:
        for obj in layout._objs:
            if isinstance(obj, LTObject):
                t.append(obj)
            else:
                t += get_text_objects(obj, ltype=ltype)
    except AttributeError:
        pass
    return t


class PDFHandler(object):
    def __init__(self, filepath, password=None):
        self.filepath = filepath
        if not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")
        if password is None:
            password = ""
        else:
            if sys.version_info[0] < 3:
                password = password.encode("ascii")

        infile = PdfFileReader(open(filepath, "rb"), strict=False)
        if infile.isEncrypted:
            infile.decrypt(password)
        self.infile = infile
        self.page_nums = infile.getNumPages()

    def write_singe_page(self, tempdir, page_num, image_path, label_path):
        p = self.infile.getPage(page_num - 1)
        pdf_page_path = os.path.join(tempdir, f"page-{page_num}.pdf")
        outfile = PdfFileWriter()
        outfile.addPage(p)
        with open(pdf_page_path, "wb") as f:
            outfile.write(f)
        layout, (width, hegiht) = get_page_layout(pdf_page_path)

        horizontal_text = get_text_objects(layout, ltype="horizontal_text")
        bboxes = [ht.bbox for ht in horizontal_text]

        if len(bboxes) < min_num:
            print("there are %d bbox, skip" % len(bboxes))
            return False

        pages = pdf2image.convert_from_path(pdf_page_path, dpi=400)
        for page in pages:
            page.save(image_path)

        new_width, new_height = pages[0].size
        scale_ratio_w = new_width / width
        scale_ratio_h = new_height / hegiht

        bboxes = np.array(bboxes) * np.array(
            [scale_ratio_w, scale_ratio_h, scale_ratio_w, scale_ratio_h,]
        )
        bboxes_ = bboxes.copy()
        bboxes_[:, 1::2] = new_height - bboxes_[:, 1::2]
        bboxes[:, 1] = bboxes_[:, 3]
        bboxes[:, 3] = bboxes_[:, 1]
        bboxes = bboxes.tolist()
        bboxes = [map(str, bbox) for bbox in bboxes]
        bboxes = [" ".join(bbox) for bbox in bboxes]
        bboxes = "\n".join(bboxes)
        with open(label_path, "w") as f:
            f.write(bboxes)
        return True

    def write_pages(self, output_dir, start_id):
        img_dir = os.path.join(output_dir, "imgs")
        label_dir = os.path.join(output_dir, "labels")
        visual_dir = os.path.join(output_dir, "visual")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(visual_dir, exist_ok=True)
        with TemporaryDirectory() as temp_dir:
            for page_num in range(1, self.page_nums + 1):
                img_path = os.path.join(img_dir, "%d_%d.png" % (start_id, page_num))
                label_path = os.path.join(label_dir, "%d_%d.txt" % (start_id, page_num))
                obj = self.write_singe_page(temp_dir, page_num, img_path, label_path)
                if not obj:
                    continue
                visual_path = os.path.join(
                    visual_dir, "%d_%d.png" % (start_id, page_num)
                )
                self.visual(img_path, label_path, visual_path)

    def visual(self, img_path, label_path, visual_path):
        img = cv2.imread(img_path)
        bboxes = open(label_path).read().split("\n")
        bboxes = [map(float, bbox.split(" ")) for bbox in bboxes]
        bboxes = [list(map(int, bbox)) for bbox in bboxes]
        for bbox in bboxes:
            cv2.rectangle(
                img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255),
            )
        cv2.imwrite(visual_path, img)


input_dir = "/home/zhoufeipeng/data/pdf"
output_dir = "/home/zhoufeipeng/data/pdf_tmp"
files = os.listdir(input_dir)
files = [os.path.join(input_dir, f) for f in files if f.lower().endswith(".pdf")]
for i, f in enumerate(files):
    try:
        pdfhandler = PDFHandler(f)
        pdfhandler.write_pages(output_dir, i)
    except:
        pass

