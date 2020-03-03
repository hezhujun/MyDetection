from datasets.coco import TCTDataset
from visualize.bbox import draw
import numpy as np
from transforms.transforms import *


categories = [
    "background",
    "ascus",
    "asch",
    "lsil",
    "hsil",
    "agc",
    "adenocarcinoma",
    "vaginalis",
    "flora/monilia",
    "dysbacteriosis"
]

if __name__ == '__main__':
    from PIL import Image
    transforms = Compose([
        Resize((1333, 800), True),
        RandomHorizontalFlip(1),
        RandomVerticalFlip(1),
    ])
    dataset = TCTDataset("/run/media/hezhujun/DATA1/Document/dataset/TCT", "tct", "/run/media/hezhujun/DATA1/Document/dataset/TCT/annotations/val.json", transforms)
    for image, labels, bboxes, scale_factor in dataset:
        pred_labels = labels
        pred_boxes = bboxes + 20
        scores = np.ones_like(labels).astype(np.float32) * 0.8
        image = draw(image, labels, bboxes, pred_labels, scores, pred_boxes, category_names=categories)
        img = Image.fromarray(image)
        img.show()
        break