import mxnet
from mxnet import nd, init, autograd
from mxnet.gluon import nn
import numpy as np

from .anchor import AnchorGenerator
from utils.coder import BoxDecoder
from utils.box import BoxClip


class RPNTargetSampler(nn.HybridBlock):

    def __init__(self, fg_thresh, bg_thresh, num_images, num_samples, pos_ratio, image_size, **kwargs):
        super(RPNTargetSampler, self).__init__(**kwargs)
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.num_images = num_images
        self.num_samples = num_samples
        self.pos_ratio = pos_ratio

        # with self.name_scope():
        #     self.box_clip = BoxClip(x_max=image_size[0], y_max=image_size[1])

    def hybrid_forward(self, F, anchors, gt_labels, gt_boxes):
        """

        :param F:
        :param anchors: (N, 4)
        :param gt_labels: (B, M)
        :param gt_boxes: (B, M, 4)
        :return:
        """
        with autograd.pause():

            # anchors = self.box_clip(anchors)

            new_indices = []
            new_matches = []
            new_mask = []

            for i in range(self.num_images):
                # (M,)
                gt_label = F.squeeze(F.slice_axis(gt_labels, axis=0, begin=i, end=i+1), axis=0)
                # (M, 4)
                gt_box = F.squeeze(F.slice_axis(gt_boxes, axis=0, begin=i, end=i+1), axis=0)

                ious = F.contrib.box_iou(anchors, gt_box, format="corner")
                ious_max = ious.max(axis=-1)
                ious_argmax = ious.argmax(axis=-1)

                # mask 1: pos 0: ignore -1: neg
                mask = F.zeros_like(ious_max)
                pos_mask = ious_max >= self.fg_thresh
                mask = F.where(pos_mask, F.ones_like(mask), mask)
                neg_mask = ious_max <= self.bg_thresh
                mask = F.where(neg_mask, F.ones_like(mask) * -1, mask)

                indices = F.argsort(F.zeros_like(mask))
                # indices = F.shuffle(indices)
                # ious_argmax = F.take(ious_argmax, indices)
                # mask = F.take(mask, indices)

                order = F.argsort(mask, is_ascend=False)
                num_pos = int(self.num_samples * self.pos_ratio)
                num_neg = self.num_samples - num_pos
                pos_indices = F.slice_axis(order, axis=0, begin=0, end=num_pos)
                neg_indices = F.slice_axis(order, axis=0, begin=-num_neg, end=None)
                sample_indices = F.concat(pos_indices, neg_indices, dim=0)

                indices = F.take(indices, sample_indices)
                ious_argmax = F.take(ious_argmax, sample_indices)
                mask = F.take(mask, sample_indices)

                new_indices.append(indices)
                new_matches.append(ious_argmax)
                new_mask.append(mask)

            return new_indices, new_matches, new_mask


class RPN(nn.HybridBlock):

    def __init__(self, channel, strides, scales, ratios, alloc_size,
                 nms_thresh, train_pre_nms, train_post_nms,
                 test_pre_nms, test_post_nms, image_size, per_level_nms=False, **kwargs):
        """

        For fpn p2, p3, p4, p5, p6
        strides = (2**2, 2**3, 2**4, 2**5, 2**6)
        scales = ((32,), (64,), (128,), (256,), (512,))
        ratios = ((0.5, 1, 2),) * len(scales)

        For c4
        strides = (2**4,)
        scales = ((128, 256, 512),)
        ratios = ((0.5, 1, 2),) * len(scales)

        :param base_size:
        :param scales:
        :param ratios:
        :param kwargs:
        """
        super(RPN, self).__init__(**kwargs)
        self.strides = strides
        self.scales = scales
        self.ratios = ratios
        num_anchor = self.num_anchors_per_layer[0]
        self.nms_thresh = nms_thresh
        self.train_pre_nms = train_pre_nms
        self.train_post_nms = train_post_nms
        self.test_pre_nms = test_pre_nms
        self.test_post_nms = test_post_nms
        self.image_size = image_size
        self.per_level_nms = per_level_nms

        az = alloc_size
        with self.name_scope():
            self.anchor_generator = nn.HybridSequential()
            for stride, scale, ratio in zip(strides, scales, ratios):
                generator = AnchorGenerator(stride, scale, ratio, az)
                self.anchor_generator.add(generator)
                az = (az[0] // 2, az[1] // 2)
            self.head = RPNHead(channel, num_anchor)
            self.box_decoder = BoxDecoder(std=(1.0, 1.0, 1.0, 1.0), convert_anchor=True)
            self.box_clip = BoxClip(x_max=image_size[0], y_max=image_size[1])

    @property
    def num_anchors_per_layer(self):
        num_layers = len(self.strides)
        assert num_layers == len(self.scales) == len(self.ratios)
        num_anchors = []
        for scale, ratio in zip(self.scales, self.ratios):
            num_scale = len(scale)
            num_ratio = len(ratio)
            num_anchor = num_scale * num_ratio
            num_anchors.append(num_anchor)

        assert all(np.array(num_anchors) == num_anchors[0])

        return tuple(num_anchors)

    def hybrid_forward(self, F, *x):
        if autograd.is_training():
            pre_nms = self.train_pre_nms
            post_nms = self.train_post_nms
        else:
            pre_nms = self.test_pre_nms
            post_nms = self.test_post_nms

        anchors = []
        rpn_pre_nms_proposals = []
        raw_rpn_scores = []
        raw_rpn_boxes = []
        for i, feat in enumerate(x):
            # raw_rpn_score (B, HWN, 1)
            # raw_rpn_box (B, HWN, 4)
            rpn_score, rpn_box, raw_rpn_score, raw_rpn_box = self.head(feat)
            with autograd.pause():
                anchor = self.anchor_generator[i](feat)
                anchor = anchor.reshape((-1, 4))  # (1, N, 4)
                anchors.append(anchor)
                # (B, N, 4)
                rpn_box = self.box_decoder(rpn_box, anchor)
                rpn_box = self.box_clip(rpn_box)
                rpn_pre = F.concat(rpn_score, rpn_box, dim=-1)
                if self.per_level_nms:
                    rpn_pre = F.contrib.box_nms(rpn_pre, overlap_thresh=self.nms_thresh, topk=pre_nms // len(x),
                                                coord_start=1, score_index=0, id_index=-1)

                rpn_pre_nms_proposals.append(rpn_pre)
                raw_rpn_scores.append(raw_rpn_score)
                raw_rpn_boxes.append(raw_rpn_box)

        rpn_pre_nms_proposals = F.concat(*rpn_pre_nms_proposals, dim=1)
        raw_rpn_scores = F.concat(*raw_rpn_scores, dim=1)
        raw_rpn_boxes = F.concat(*raw_rpn_boxes, dim=1)

        with autograd.pause():
            if self.per_level_nms:
                # Sort the proposals by scores. So the overlap_thresh=2
                tmp = F.contrib.box_nms(rpn_pre_nms_proposals, overlap_thresh=2, topk=pre_nms + 1, coord_start=1,
                                        score_index=0, id_index=-1)
            else:
                tmp = F.contrib.box_nms(rpn_pre_nms_proposals, overlap_thresh=self.nms_thresh, topk=pre_nms,
                                        coord_start=1, score_index=0, id_index=-1)

        result = F.slice_axis(tmp, axis=1, begin=0, end=post_nms)
        rpn_scores = F.slice_axis(result, axis=-1, begin=0, end=1)
        rpn_boxes = F.slice_axis(result, axis=-1, begin=1, end=None)

        if autograd.is_training():
            return rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes, anchors
        else:
            return rpn_scores, rpn_boxes

    @property
    def is_multi_layers(self):
        return len(self.strides) > 1


class RPNHead(nn.HybridBlock):

    def __init__(self, channel, num_anchor, **kwargs):
        super(RPNHead, self).__init__(**kwargs)
        weight_initializer = init.Normal(0.01)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            self.conv.add(nn.Conv2D(channel, 3, 1, 1, weight_initializer=weight_initializer),
                          nn.Activation('relu'))
            self.cls = nn.Conv2D(num_anchor, 1, 1, 0, weight_initializer=weight_initializer)
            self.reg = nn.Conv2D(num_anchor * 4, 1, 1, 0, weight_initializer=weight_initializer)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        # (B, C, H, W)->(B, N, H, W)->(B, H, W, N)->(B, HWN, 1)
        raw_rpn_scores = self.cls(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 1))
        # (B, HWN, 1)
        rpn_scores = F.sigmoid(F.stop_gradient(raw_rpn_scores))
        # (B, C, H, W)->(B, N*4, H, W)->(B, H, W, N*4)->(B, H, W, N, 4)->(B, HWN, 4)
        raw_rpn_boxes = self.reg(x).transpose(axes=(0, 2, 3, 1)).reshape((0, 0, 0, -1, 4)).reshape((0, -1, 4))
        # (B, HWN, 4)
        rpn_boxes = F.stop_gradient(raw_rpn_boxes)
        return rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes
