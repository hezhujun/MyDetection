from datasets.coco import TCTDataset
from transforms.transforms import *
from mxnet.gluon.data import DataLoader
from mxnet import nd, autograd, gluon, gpu, init
from mxnet.gluon import utils as gutils
from visualize.bbox import draw
from PIL import Image
from utils.data_pipline import collate_fn
from network import FasterRCNNDetector
import os
import sys
from engine import inference
from pycocotools.cocoeval import COCOeval
import json


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["MXNET_ENABLE_GPU_P2P"] = "0"
    gpus = [0, 1, 2, 3]
    is_multi_gpus = len(gpus) > 1
    images_per_gpu = 1
    batch_size = images_per_gpu * len(gpus)
    num_workers = images_per_gpu * len(gpus)

    root = "/root/userfolder/datasets/TCT"
    # root = "/run/media/hezhujun/DATA1/Document/dataset/TCT"
    val_ann_file = os.path.join(root, "annotations/val.json")
    val_transforms = Compose([
        Resize((1333, 800), True),
        Normalize(mean=(127, 127, 127), std=(255, 255, 255)),
        ToTensor()
    ])
    val_dataset = TCTDataset(root, "tct_val", val_ann_file, val_transforms)
    val_dataset.ids = val_dataset.ids[0:40]
    val_data_loader = DataLoader(val_dataset, batch_size, False, last_batch="keep", batchify_fn=collate_fn, num_workers=num_workers)

    anchor_scales = {
        "p2": (32,),
        "p3": (64,),
        "p4": (128,),
        "p5": (256,),
        "p6": (512,),
    }
    anchor_ratios = {
        "p2": ((1, 2, 0.5),),
        "p3": ((1, 2, 0.5),),
        "p4": ((1, 2, 0.5),),
        "p5": ((1, 2, 0.5),),
        "p6": ((1, 2, 0.5),),
    }

    if is_multi_gpus:
        device = [gpu(i) for i in gpus]
    else:
        device = gpu(gpus[0])

    detector = FasterRCNNDetector(val_dataset.num_classes, anchor_scales, anchor_ratios, use_fpn=True, ctx=device)
    if detector.use_fpn:
        detector.fpn.initialize(init=init.Xavier(), ctx=device)
    detector.rpn.initialize(init=init.Xavier(), ctx=device)
    detector.roi_extractor.initialize(init=init.Xavier(), ctx=device)
    detector.head.initialize(init=init.Xavier(), ctx=device)
    detector.cls.initialize(init=init.Xavier(), ctx=device)
    detector.reg.initialize(init=init.Xavier(), ctx=device)
    detector.cls_loss.initialize(init=init.Xavier(), ctx=device)
    detector.reg_loss.initialize(init=init.Xavier(), ctx=device)

    detector.hybridize()

    print("Begin...")
    results = inference(detector, val_data_loader, device)
    print("End...")

    print(json.dumps(results, indent=2))

    if len(results) > 0:
        cocoGT = val_data_loader._dataset.coco
        imgIds = val_data_loader._dataset.ids
        cocoDT = cocoGT.loadRes(results)
        cocoEval = COCOeval(cocoGT, cocoDT, "bbox")
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
