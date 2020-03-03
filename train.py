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


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
    os.environ["MXNET_ENABLE_GPU_P2P"] = "0"
    gpus = [0, 1, 2, 3]
    is_multi_gpus = len(gpus) > 1
    images_per_gpu = 1
    batch_size = images_per_gpu * len(gpus)
    num_workers = images_per_gpu * len(gpus)

    root = "/root/userfolder/datasets/TCT"
    # root = "/run/media/hezhujun/DATA1/Document/dataset/TCT"
    train_ann_file = os.path.join(root, "annotations/train.json")
    val_ann_file = os.path.join(root, "annotations/val.json")
    test_ann_file = os.path.join(root, "annotations/test.json")
    train_transforms = Compose([
        Resize((1333, 800), True),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        Normalize(mean=(127, 127, 127), std=(255, 255, 255)),
        ToTensor()
    ])
    val_transforms = Compose([
        Resize((1333, 800), True),
        Normalize(mean=(127, 127, 127), std=(255, 255, 255)),
        ToTensor()
    ])
    train_dataset = TCTDataset(root, "tct_train", train_transforms)
    train_data_loader = DataLoader(train_dataset, batch_size, True, last_batch="rollover", batchify_fn=collate_fn,
                                 num_workers=num_workers)
    val_dataset = TCTDataset(root, "tct_val", val_transforms)
    val_data_loader = DataLoader(val_dataset, batch_size, False, last_batch="discard", batchify_fn=collate_fn, num_workers=num_workers)

    # anchor_scales = {
    #     "c4": (64, 128, 256),
    # }
    # anchor_ratios = {
    #     "c4": ((1, 2, 0.5),) * 3,
    # }

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

    trainer = gluon.Trainer(detector.collect_params(), 'sgd', {'learning_rate': 0.001, "wd": 1e-5})

    num_epochs = 5
    for epoch in range(num_epochs):
        iteration = 0
        for images, labels, bboxes, scale_factors, image_ids in train_data_loader:
            if is_multi_gpus:
                images = gutils.split_and_load(images, device)
                labels = gutils.split_and_load(labels, device)
                bboxes = gutils.split_and_load(bboxes, device)
                scale_factors = gutils.split_and_load(scale_factors, device)

                rpn_cls_losses = []
                rpn_reg_losses = []
                cls_lossse = []
                reg_lossse = []
                total_losses = []
                with autograd.record():
                    for images_, labels_, bboxes_, scale_factors_ in zip(images, labels, bboxes, scale_factors):
                        rpn_cls_loss, rpn_reg_loss, cls_loss, reg_loss = detector(images_, scale_factors_, labels_, bboxes_)
                        rpn_cls_losses.append(rpn_cls_loss)
                        rpn_reg_losses.append(rpn_reg_loss)
                        cls_lossse.append(cls_loss)
                        reg_lossse.append(reg_loss)
                        total_loss = sum([rpn_cls_loss, rpn_reg_loss, cls_loss, reg_loss])
                        total_losses.append(total_loss)

                for total_loss in total_losses:
                    total_loss.backward()
                rpn_cls_loss = np.mean([loss.asscalar() for loss in rpn_cls_losses])
                rpn_reg_loss = np.mean([loss.asscalar() for loss in rpn_reg_losses])
                cls_loss = np.mean([loss.asscalar() for loss in cls_lossse])
                reg_loss = np.mean([loss.asscalar() for loss in reg_lossse])
                total_loss = np.mean([loss.asscalar() for loss in total_losses])
            else:
                images = images.as_in_context(device)
                labels = labels.as_in_context(device)
                bboxes = bboxes.as_in_context(device)
                scale_factors = scale_factors.as_in_context(device)

                with autograd.record():
                    rpn_cls_loss, rpn_reg_loss, cls_loss, reg_loss = detector(images, scale_factors, labels, bboxes)
                    total_loss = sum([rpn_cls_loss, rpn_reg_loss, cls_loss, reg_loss])

                total_loss.backward()
                rpn_cls_loss = rpn_cls_loss.asscalar()
                rpn_reg_loss = rpn_reg_loss.asscalar()
                cls_loss = cls_loss.asscalar()
                reg_loss = reg_loss.asscalar()
                total_loss = total_loss.asscalar()

            trainer.step(1)
            nd.waitall()

            print("epoch {}, iter {}/{}, rpn_cls_loss {}, rpn_reg_loss {}, cls_loss {}, reg_loss {} total_loss {}".format(
                epoch, iteration, len(train_data_loader), rpn_cls_loss, rpn_reg_loss, cls_loss, reg_loss, total_loss))

            iteration += 1

        print("Evaluate...")
        results = inference(detector, val_data_loader, device)
        if len(results) > 0:
            cocoGT = val_data_loader._dataset.coco
            imgIds = val_data_loader._dataset.ids
            cocoDT = cocoGT.loadRes(results)
            cocoEval = COCOeval(cocoGT, cocoDT, "bbox")
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
