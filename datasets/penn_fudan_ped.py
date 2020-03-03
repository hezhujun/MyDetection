import os

import cv2
import numpy as np
from mxnet.gluon.data import Dataset
from gluoncv.data.mscoco.detection import COCODetection


class PennFudanDataset(COCODetection):
    CLASSES = [
        "PASpersonWalking",
    ]

    def __init__(self, root, annFile, transform=None, min_object_area=0,
                 skip_empty=True, use_crowd=True):
        super(PennFudanDataset, self).__init__(root, annFile, transform, min_object_area, skip_empty, use_crowd)

    def _parse_image_path(self, entry):
        image_path = entry["file_name"]
        return os.path.join(self._root, image_path)
