import os

import cv2
import numpy as np
from mxnet.gluon.data import Dataset
from gluoncv.data.mscoco.detection import COCODetection
from pycocotools.coco import COCO


class COCODataset(object):

    CLASSES = [
        "PASpersonWalking",
    ]

    def __init__(self, root, ann_file, transform=None, return_image=True):
        self.root = root
        self.ann_file = ann_file
        self.transform = transform
        self.return_image = return_image

        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = self.coco.loadImgs(img_id)[0]
        img_path = self.parse_image_path(self.root, img)
        if self.return_image:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_path
        labels = []
        bboxes = []
        scale_factor = [1, 1, 1, 1]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            labels.append(ann["category_id"])
            x1, y1, w, h = ann["bbox"]
            bboxes.append([x1, y1, x1 + w, y1 + h])

        if self.transform is not None:
            return self.transform(img, labels, bboxes, scale_factor, img_id)
        return img, labels, bboxes, scale_factor, img_id

    def parse_image_path(self, root, item):
        return os.path.join(root, item["file_name"])


# class TCTDataset(COCODetection):
#     CLASSES = [
#         "background",
#         "ascus",
#         "asch",
#         "lsil",
#         "hsil",
#         "agc",
#         "adenocarcinoma",
#         "vaginalis",
#         "flora/monilia",
#         "dysbacteriosis",
#     ]
#
#     def __init__(self, root, annFile, transform=None, min_object_area=0,
#                  skip_empty=True, use_crowd=True):
#         super(TCTDataset, self).__init__(root, annFile, transform, min_object_area, skip_empty, use_crowd)
#
#     def _parse_image_path(self, entry):
#         image_path = entry["file_name"]
#         return os.path.join(self._root, image_path)
