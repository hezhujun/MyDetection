import os
import mxnet as mx
from mxnet import nd, cpu, gpu, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
import itertools
import numpy as np
from pycocotools.cocoeval import COCOeval

from model.faster_rcnn.faster_rcnn import FasterRCNNDetector
from model.resnet.resnet import ResNet50
from model.fpn.fpn import FPN
from mxnet.gluon.data import DataLoader
import mxnet.gluon.utils as gutils

from transforms.transforms import *
from datasets.coco import COCODataset
from utils.data_pipline import Collator
from gluoncv.model_zoo.resnetv1b import resnet50_v1b
from gluoncv.nn.feature import FPNFeatureExpander
from mxnet.gluon.contrib.nn import SyncBatchNorm


def multi_gpus_forward(net, ctx, img, labels=None, bboxes=None):
    img_list = gutils.split_and_load(img, ctx)

    if autograd.is_training():
        labels_list = gutils.split_and_load(labels, ctx)
        bboxes_list = gutils.split_and_load(bboxes, ctx)

        rpn_cls_losses = []
        rpn_box_losses = []
        rcnn_cls_losses = []
        rcnn_box_losses = []
        total_losses = []
        for img, labels, bboxes in zip(img_list, labels_list, bboxes_list):
            rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss, total_loss = net(img, labels, bboxes)
            rpn_cls_losses.append(rpn_cls_loss)
            rpn_box_losses.append(rpn_box_loss)
            rcnn_cls_losses.append(rcnn_cls_loss)
            rcnn_box_losses.append(rcnn_box_loss)
            total_losses.append(total_loss)

        rpn_cls_loss = np.mean([loss.asscalar() for loss in rpn_cls_losses])
        rpn_box_loss = np.mean([loss.asscalar() for loss in rpn_box_losses])
        rcnn_cls_loss = np.mean([loss.asscalar() for loss in rpn_cls_losses])
        rcnn_box_loss = np.mean([loss.asscalar() for loss in rcnn_cls_losses])
        total_loss = np.mean([loss.asscalar() for loss in total_losses])
        return rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss, total_loss

    else:
        cls_ids_list = []
        scores_list = []
        bboxes_list = []
        for img in img_list:
            cls_ids, scores, bboxes = net(img)
            cls_ids_list.append(cls_ids)
            scores_list.append(scores)
            bboxes_list.append(bboxes)

        cls_ids_list = [item.asnumpy() for item in cls_ids_list]
        scores_list = [item.asnumpy() for item in scores_list]
        bboxes_list = [item.asnumpy() for item in bboxes_list]
        cls_ids = np.concatenate(cls_ids_list)
        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        return cls_ids, scores, bboxes


class TrainTask(object):

    def __init__(self, net, **kwargs):
        self.net = net
        self.rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self.rpn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1
        self.rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1

    def __call__(self, x, gt_labels=None, gt_boxes=None):
        with autograd.record():
            # raw_rpn_scores (B, rpn_num_samples, 1)
            # rpn_masks (B, rpn_num_samples) 1: pos 0: ignore -1: neg
            # raw_rpn_boxes (B, rpn_num_samples, 4)
            # raw_rpn_box_targets (B, rpn_num_samples, 4)
            # cls_pred (B, num_samples, C+1)
            # cls_targets (B, num_samples) value [-1, C) -1: ignore 0: background
            # box_pred (B, num_pos, 4)
            # reg_targets (B, num_pos, 4)
            # box_mask (B, num_pos, 4)
            raw_rpn_scores, rpn_masks, raw_rpn_boxes, raw_rpn_box_targets, cls_pred, cls_targets, box_pred, \
            reg_targets, box_mask = self.net(x, gt_labels, gt_boxes)

            raw_rpn_scores = nd.squeeze(raw_rpn_scores, axis=-1)
            rpn_score_targets = nd.where(rpn_masks == 1, rpn_masks, nd.zeros_like(rpn_masks))
            raw_rpn_scores = nd.reshape(raw_rpn_scores, (-1,))
            rpn_score_targets = nd.reshape(rpn_score_targets, (-1,))
            rpn_masks = nd.reshape(rpn_masks, (-1,))
            raw_rpn_boxes = nd.reshape(raw_rpn_boxes, (-1, 4))
            raw_rpn_box_targets = nd.reshape(raw_rpn_box_targets, (-1, 4))
            num_samples = (rpn_masks != 0).sum()
            rpn_cls_loss = self.rpn_cls_loss(raw_rpn_scores, rpn_score_targets, rpn_masks != 0)
            rpn_cls_loss = rpn_cls_loss.sum() / num_samples
            rpn_masks = nd.reshape(rpn_masks, (-1, 1))
            rpn_box_loss = self.rpn_box_loss(raw_rpn_boxes, raw_rpn_box_targets, rpn_masks == 1)
            rpn_box_loss = rpn_box_loss.sum() / num_samples

            B, num_samples, C_1 = cls_pred.shape
            cls_pred = nd.reshape(cls_pred, (-1, C_1))
            cls_targets = nd.reshape(cls_targets, (-1,))
            box_pred = nd.reshape(box_pred, (-1, 4))
            reg_targets = nd.reshape(reg_targets, (-1, 4))
            box_mask = nd.reshape(box_mask, (-1, 4))
            num_samples = (cls_targets >= 0).sum()
            rcnn_cls_loss = self.rcnn_cls_loss(cls_pred, cls_targets, cls_targets >= 0)
            rcnn_cls_loss = rcnn_cls_loss.sum() / num_samples
            rcnn_box_loss = self.rcnn_box_loss(box_pred, reg_targets, box_mask == 1)
            rcnn_box_loss = rcnn_box_loss.sum() / num_samples

            total_loss = rpn_cls_loss + rpn_box_loss + rcnn_cls_loss + rcnn_box_loss

            total_loss.backward()

        return rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss, total_loss


def validate(net, data_loader, ctx):
    # input format is differnet than training, thus rehybridization is needed.
    net.hybridize()
    results = []
    for img, labels, bboxes, scale_factor, img_id in data_loader:
        if isinstance(ctx, (list, tuple)):
            cls_ids, scores, bboxes = multi_gpus_forward(net, ctx, img)
        else:
            img = img.as_in_context(ctx)

            cls_ids, scores, bboxes = net(img)
            cls_ids = cls_ids.asnumpy()
            scores = scores.asnumpy()
            bboxes = bboxes.asnumpy()

        batch_size, num_objs = cls_ids.shape[0], cls_ids.shape[1]
        for i in range(batch_size):
            for j in range(num_objs):
                cls_id = int(cls_ids[i, j, 0])
                if cls_id == -1:
                    break
                x1, y1, x2, y2 = bboxes[i, j, :]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                results.append({
                    "image_id": int(img_id[i]),
                    "category_id": cls_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(scores[i, j, 0])
                })
        # break

    coco = data_loader._dataset.coco
    imgIds = data_loader._dataset.ids
    cocoDT = coco.loadRes(results)
    cocoEval = COCOeval(coco, cocoDT, "bbox")
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':

    batch_size = 2
    num_workers = 4
    image_size = (512, 512)
    num_epochs = 5

    root = "E:\\dataset\\PennFudanPed"
    train_ann_file = os.path.join(root, "annotations\\PennFudanPed.json")
    val_ann_file = os.path.join(root, "annotations\\PennFudanPed.json")
    train_transforms = Compose([
        Resize(image_size, True),
        RandomHorizontalFlip(),
        Normalize(mean=(127, 127, 127), std=(255,255, 255)),
        ToTensor()
    ])
    val_transforms = Compose([
        Resize(image_size, True),
        Normalize(mean=(127, 127, 127), std=(255, 255, 255)),
        ToTensor()
    ])

    train_dataset = COCODataset(root, train_ann_file, train_transforms)
    train_data_loader = DataLoader(train_dataset, batch_size, True, last_batch="discard", batchify_fn=Collator(10), num_workers=num_workers)

    val_dataset = COCODataset(root, val_ann_file, val_transforms)
    val_data_loader = DataLoader(val_dataset, batch_size, False, last_batch="discard", batchify_fn=Collator(10), num_workers=num_workers)

    ctx = cpu()
    num_devices = 1
    gluon_norm_kwargs = {"num_devices": num_devices} if num_devices >= 1 else {}
    base_network = resnet50_v1b(pretrained=True, dilated=False, use_global_stats=False,
                                norm_layer=SyncBatchNorm, norm_kwargs=gluon_norm_kwargs)
    sym_norm_kwargs = {"ndev": num_devices} if num_devices >= 1 else {}
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd', 'layers4_relu8_fwd'],
        num_filters=[256, 256, 256, 256], use_1x1=True, use_upsample=True, use_elewadd=True, use_p6=True,
        no_bias=True, pretrained=True, norm_layer=mx.sym.contrib.SyncBatchNorm, norm_kwargs=sym_norm_kwargs
    )
    box_features = nn.HybridSequential()
    box_features.add(nn.Conv2D(256, 3, padding=1, use_bias=False),
                     SyncBatchNorm(**gluon_norm_kwargs),
                     nn.Activation('relu'),
                     nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                     nn.Activation('relu'))

    # resnet50 = vision.resnet50_v1(pretrained=True, ctx=ctx).features
    # backbone = nn.HybridSequential()
    # for i in [i for i in range(7)]:
    #     backbone.add(resnet50[i])
    #
    # head = nn.HybridSequential()
    # head.add(resnet50[7])
    # head.add(nn.GlobalAvgPool2D())

    # strides = (2 ** 4,)
    # scales = ((64**2, 128**2, 256**2, 512**2),)
    # ratios = ((0.5, 1, 2),) * len(scales)
    # alloc_size = (512, 512)

    strides = (2**2,2**3,2**4,2**5,2**6)
    scales = ((32**2,), (64**2,), (128**2,), (256**2,), (512**2,))
    ratios = ((0.5, 1, 2),) * len(scales)
    alloc_size = (512, 512)

    detector = FasterRCNNDetector(features, box_features, num_classes=1, strides=strides, scales=scales, ratios=ratios,
                                  batch_size_per_device=batch_size, image_size=image_size)
    for param in detector.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()
    detector.collect_params().reset_ctx(ctx)
    train_task = TrainTask(detector)
    # (self.head.collect_params, self.rpn.collect_params(), self.cls.collect_params(), self.reg.collect_params())
    trainer = gluon.Trainer(detector.collect_params(), 'sgd', {'learning_rate': 0.001, "wd": 1e-5})
    print("Begin")
    for epoch in range(num_epochs):
        detector.hybridize()
        for img, labels, bboxes, scale_factor, img_id in train_data_loader:
            if isinstance(ctx, (list, tuple)):
                rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss, total_loss = \
                    multi_gpus_forward(train_task, ctx, img, labels, bboxes)
            else:
                img = img.as_in_context(ctx)
                labels = labels.as_in_context(ctx)
                bboxes = bboxes.as_in_context(ctx)
                scale_factor = scale_factor.as_in_context(ctx)

                rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss, total_loss = train_task(img, labels, bboxes)
                rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss, total_loss = \
                    rpn_cls_loss.asscalar(), rpn_box_loss.asscalar(), rcnn_cls_loss.asscalar(), \
                    rcnn_box_loss.asscalar(), total_loss.asscalar()
                trainer.step(batch_size)

            print("epoch {} rpn_cls_loss {} rpn_box_loss {} rcnn_cls_loss {} rcnn_box_loss {} total_loss {}".format(
                epoch,
                rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss, total_loss
            ))
            # break
        print("Evaluate")
        validate(detector, val_data_loader, ctx)
    exit()

    # ctx = cpu()
    # x = nd.random_normal(loc=0, scale=0.1, shape=(2, 3, 800, 600), ctx=ctx)
    # gt_labels = nd.array([[1, 3, 2],
    #                       [2, 1, -1]])
    # gt_boxes = nd.array([[[10, 10, 300, 400],[310, 10, 600, 380],[200, 400, 500, 790]],
    #                      [[10, 15, 300, 300],[310, 310, 580, 800],[-1, -1, -1, -1]]])
    #
    # # resnet50 = ResNet50(output_layers=[4,], pretrained=True, ctx=ctx)


