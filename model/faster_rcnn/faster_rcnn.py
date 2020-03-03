import mxnet
from mxnet import nd, autograd, init
from mxnet.gluon import nn

from utils.box import BoxClip
from ..rpn.rpn import RPN, RPNTargetSampler
from utils.coder import BoxEncoder, BoxDecoder


class RCNNTargetSampler(nn.HybridBlock):

    def __init__(self, num_image, num_sample, pos_iou_thresh, pos_ratio, num_proposal, max_num_gt, **kwargs):
        super(RCNNTargetSampler, self).__init__(**kwargs)
        self.num_image = num_image
        self.num_sample = num_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.pos_ratio = pos_ratio
        self.max_pos = int(round(num_sample * pos_ratio))
        self.num_proposal = num_proposal
        self.max_num_gt = max_num_gt

    def hybrid_forward(self, F, scores, boxes, gt_scores, gt_boxes):
        """

        :param F:
        :param scores: (B, N, 1) -1 means ignore
        :param boxes: (B, N, 4)
        :param gt_scores: (B, M) -1 means ignore
        :param gt_boxes: (B, M, 4)
        :return:
        new_boxes: (B, num_samples, 4)
        new_pos_neg: (B, num_samples)  -1: neg 0: ignore 1: pos
        new_matches:  (B, num_samples)  matched index
        """
        with autograd.pause():
            new_boxes = []
            new_pos_neg = []
            new_matches = []
            for i in range(self.num_image):
                # (N, 4)
                box = F.squeeze(F.slice_axis(boxes, axis=0, begin=i, end=i + 1), axis=0)
                # (N, )
                score = F.squeeze(F.slice_axis(scores, axis=0, begin=i, end=i + 1), axis=(0,2))
                # (M, 4)
                gt_box = F.squeeze(F.slice_axis(gt_boxes, axis=0, begin=i, end=i + 1), axis=0)
                # (M, )
                gt_score = F.squeeze(F.slice_axis(gt_scores, axis=0, begin=i, end=i + 1), axis=0)

                # concat proposals with ground truth
                gt_score = F.sign(gt_score)
                all_box = F.concat(box, gt_box, dim=0)
                all_score = F.concat(score, gt_score, dim=0)

                ious = F.contrib.box_iou(all_box, gt_box, format="corner")
                ious_max = ious.max(axis=-1)
                ious_argmax = ious.argmax(axis=-1)
                # init with 2. 2: neg samples
                mask = F.ones_like(ious_max) * 2
                # 0: ignore
                mask = F.where(all_score < 0, F.zeros_like(mask), mask)
                # 3: pos samples
                pos_mask = ious_max >= self.pos_iou_thresh
                mask = F.where(pos_mask, F.ones_like(mask) * 3, mask)

                # shuffle mask
                rand = F.random_uniform(0, 1, shape=(self.num_proposal + self.max_num_gt,))
                rand = F.slice_like(rand, ious_argmax)
                index = F.argsort(rand)
                mask = F.take(mask, index)
                ious_argmax = F.take(ious_argmax, index)

                # sample pos samples
                order = F.argsort(mask, is_ascend=False)
                topk = F.slice_axis(order, axis=0, begin=0, end=self.max_pos)
                topk_indices = F.take(index, topk)
                topk_pos_neg = F.take(mask, topk)
                topk_matches = F.take(ious_argmax, topk)
                # reset output: 3 pos 2 neg 0 ignore -> 1 pos -1 neg 0 ignore
                topk_pos_neg = F.where(topk_pos_neg == 3, F.ones_like(topk_pos_neg), topk_pos_neg)
                topk_pos_neg = F.where(topk_pos_neg == 2, F.ones_like(topk_pos_neg) * -1, topk_pos_neg)

                # sample neg samples
                last = F.slice_axis(order, axis=0, begin=self.max_pos, end=None)
                index = F.take(index, last)
                mask = F.take(mask, last)
                ious_argmax = F.take(ious_argmax, last)
                # change mask: 4 neg 3 pos 0 ignore
                mask = F.where(mask == 2, F.ones_like(mask) * 4, mask)
                order = F.argsort(mask, is_ascend=False)
                num_neg = self.num_sample - self.max_pos
                bottomk = F.slice_axis(order, axis=0, begin=0, end=num_neg)
                bottomk_indices = F.take(index, bottomk)
                bottomk_pos_neg = F.take(mask, bottomk)
                bottomk_matches = F.take(ious_argmax, bottomk)
                # reset outpu: 3 pos 4 neg 0 ignore -> 1 pos -1 neg 0 ignore
                bottomk_pos_neg = F.where(bottomk_pos_neg == 3, F.ones_like(bottomk_pos_neg), bottomk_pos_neg)
                bottomk_pos_neg = F.where(bottomk_pos_neg == 4, F.ones_like(bottomk_pos_neg) * -1, bottomk_pos_neg)

                # output
                indices = F.concat(topk_indices, bottomk_indices, dim=0)
                pos_neg = F.concat(topk_pos_neg, bottomk_pos_neg, dim=0)
                matches = F.concat(topk_matches, bottomk_matches, dim=0)

                sampled_boxes = all_box.take(indices)
                new_boxes.append(sampled_boxes)
                new_pos_neg.append(pos_neg)
                new_matches.append(matches)
            new_boxes = F.stack(*new_boxes, axis=0)
            new_pos_neg = F.stack(*new_pos_neg, axis=0)
            new_matches = F.stack(*new_matches, axis=0)
            return new_boxes, new_pos_neg, new_matches


class FasterRCNNDetector(nn.HybridBlock):
    def __init__(self, backbone, head, num_classes, strides, scales, ratios, batch_size_per_device,
                 image_size, nms_thresh=0.3, nms_topk=400, post_nms=100, force_nms=False, rpn_channel=1024,
                 alloc_size=(256, 256), rpn_nms_thresh=0.7,
                 rpn_train_pre_nms=12000, rpn_train_post_nms=2000,  rpn_test_pre_nms=6000, rpn_test_post_nms=300,
                 rpn_fg_thresh=0.5, rpn_bg_thresh=0.3, rpn_num_samples=512, rpn_pos_ratio=0.25,
                 nms_per_layer=False,
                 roi_mode="align", roi_size=(14, 14), num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
                 max_num_gt=300, **kwargs):
        super(FasterRCNNDetector, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.force_nms = force_nms
        self.post_nms = post_nms
        self.roi_mode = roi_mode
        self.roi_size = roi_size
        self.num_sample = num_sample
        self.rpn_test_post_nms = rpn_test_post_nms
        self.batch_size = batch_size_per_device
        self.strides = strides
        self.pos_ratio = pos_ratio
        self.max_pos = int(round(num_sample * pos_ratio))
        with self.name_scope():
            self.backbone = backbone
            self.head = head
            self.rpn = RPN(rpn_channel, strides, scales, ratios, alloc_size, rpn_nms_thresh, rpn_train_pre_nms,
                           rpn_train_post_nms, rpn_test_pre_nms, rpn_test_post_nms, image_size, nms_per_layer)
            self.rpn_sampler = RPNTargetSampler(rpn_fg_thresh, rpn_bg_thresh, batch_size_per_device,
                                                rpn_num_samples, rpn_pos_ratio, image_size)
            self.sampler = RCNNTargetSampler(batch_size_per_device, num_sample, pos_iou_thresh, pos_ratio,
                                             rpn_train_post_nms, max_num_gt)
            self.cls = nn.Dense(self.num_classes + 1, weight_initializer=init.Normal(0.01))
            self.reg = nn.Dense(self.num_classes * 4, weight_initializer=init.Normal(0.001))
            self.box_encoder = BoxEncoder(std=(1.0, 1.0, 1.0, 1.0), convert_anchor=True, convert_box=True)
            self.box_decoder = BoxDecoder(std=(1.0, 1.0, 1.0, 1.0), convert_anchor=True)
            self.box_clip = BoxClip(x_max=image_size[0], y_max=image_size[1])

    def hybrid_forward(self, F, x, gt_labels=None, gt_boxes=None):
        """

        :param F:
        :param x:
        :param gt_labels:
        :param gt_boxes:
        :return:
            cls_pred (B, num_sample, C+1)
            box_pred (B, max_pos, C, 4)
            rpn_boxes (B, num_sample, 4)
            pos_neg_mask (B, num_sample)  1: pos 0: ignore -1: neg
            matches (B, N) value in [0, num_gt)
            raw_rpn_scores (B, -1, 1)
            raw_rpn_boxes (B, -1, 4)
            anchors List [(1, HW*num_anchor_per_pixel, 4)]
            cls_targets (B, num_sample)
            reg_targets (B, max_pos, 4)
            box_mask (B, max_pos, 4)
        """
        feat = self.backbone(x)
        if not isinstance(feat, (list, tuple)):
            feat = [feat]

        if autograd.is_training():
            # rpn_scores ()
            # rpn_scores ()
            # raw_rpn_scores (B, N, 1)
            # raw_rpn_boxes (B, N, 4)
            # anchors list each (?, 4)
            rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes, anchors = self.rpn(*feat)

            anchors = F.concat(*anchors, dim=0)
            # rpn_sample_indices list each (num_rpn_samples, )
            # rpn_matches list each (num_rpn_samples, )
            # rpn_masks list each (num_rpn_samples, )
            rpn_sample_indices, rpn_matches, rpn_masks = self.rpn_sampler(anchors, gt_labels, gt_boxes)
            with autograd.pause():
                raw_rpn_box_targets = []
                for i in range(self.batch_size):
                    rpn_sample_index = rpn_sample_indices[i]
                    # (rpn_num_samples, 4)
                    rpn_anchor = F.take(anchors, rpn_sample_index)
                    # (M, 4)
                    rpn_box_target = F.squeeze(F.slice_axis(gt_boxes, axis=0, begin=i, end=i+1), axis=0)
                    rpn_match = rpn_matches[i]
                    # (rpn_num_samples, 4)
                    rpn_box_target = F.take(rpn_box_target, rpn_match)
                    rpn_raw_box_target = self.box_encoder(rpn_box_target, rpn_anchor)
                    raw_rpn_box_targets.append(rpn_raw_box_target)

                # (B, num_rpn_samples, 4)
                raw_rpn_box_targets = F.stack(*raw_rpn_box_targets, axis=0)
                raw_rpn_box_targets = F.stop_gradient(raw_rpn_box_targets)
                # (B, num_rpn_samples)
                rpn_masks = F.stack(*rpn_masks, axis=0)
                rpn_masks = F.stop_gradient(rpn_masks)

            new_raw_rpn_scores = []
            new_raw_rpn_boxes = []
            for i in range(self.batch_size):
                raw_rpn_score = F.squeeze(F.slice_axis(raw_rpn_scores, axis=0, begin=i, end=i+1), axis=0)
                raw_rpn_box = F.squeeze(F.slice_axis(raw_rpn_boxes, axis=0, begin=i, end=i+1), axis=0)
                rpn_sample_index = rpn_sample_indices[i]
                raw_rpn_score = F.take(raw_rpn_score, rpn_sample_index)
                raw_rpn_box = F.take(raw_rpn_box, rpn_sample_index)
                new_raw_rpn_scores.append(raw_rpn_score)
                new_raw_rpn_boxes.append(raw_rpn_box)

            raw_rpn_scores = F.stack(*new_raw_rpn_scores, axis=0)
            raw_rpn_boxes = F.stack(*new_raw_rpn_boxes, axis=0)
            rpn_masks = F.stop_gradient(rpn_masks)
            raw_rpn_box_targets = F.stop_gradient(raw_rpn_box_targets)

            rpn_boxes, pos_neg_mask, matches = self.sampler(rpn_scores, rpn_boxes, gt_labels, gt_boxes)
        else:
            rpn_scores, rpn_boxes = self.rpn(*feat)

        # create batch_id for proposals
        num_proposals = self.num_sample if autograd.is_training() else self.rpn_test_post_nms
        with autograd.pause():
            batch_id = F.arange(0, self.batch_size)
            batch_id = F.repeat(batch_id, num_proposals)
            rpn_roi = F.concat(*[batch_id.reshape((-1,1)), rpn_boxes.reshape((-1, 4))], dim=-1)
            rpn_roi = F.stop_gradient(rpn_roi)

        if len(feat) > 1:
            pooled_feat = self._pyramid_roi_feats(F, feat, rpn_roi)
        else:
            if self.roi_mode == "pool":
                pooled_feat = F.ROIPooling(feat[0], rpn_roi, self.roi_size, 1.0 / self.strides[0])
            elif self.roi_mode == "align":
                pooled_feat = F.contrib.ROIAlign(feat[0], rpn_roi, self.roi_size, 1.0 / self.strides[0], sample_ratio=2)
            else:
                raise ValueError("Invalid roi mode: {}".format(self.roi_mode))

        # (BN, channels, 1, 1)
        roi_feat = self.head(pooled_feat)

        # (BN, C+1)
        cls_pred = self.cls(roi_feat)
        # (BN, C+1) -> (B, N, C+1)
        cls_pred = cls_pred.reshape((self.batch_size, -1, self.num_classes + 1))

        if autograd.is_training():
            with autograd.pause():
                # (B, M) -> (B, 1, M) -> (B, N, M)
                gt_labels = F.broadcast_like(F.reshape(gt_labels, (0, 1, -1)), matches, lhs_axes=1, rhs_axes=1)
                # ids (B, N, M) -> (B, N), value [0, C+1), 0 reserved for background clss
                target_ids = F.pick(gt_labels, matches, axis=2) + 1   # the gt_label may be -1
                # set ignore samples to -1
                cls_targets = F.where(pos_neg_mask == 1, target_ids, F.ones_like(target_ids) * -1)
                # set negative samples to 0
                cls_targets = F.where(pos_neg_mask == -1, F.zeros_like(cls_targets), cls_targets)
                cls_targets = F.stop_gradient(cls_targets)

                # (B, M, 4) -> (B, 1, M, 4) -> (B, N, M, 4)
                gt_boxes = F.broadcast_like(F.reshape(gt_boxes, (0, 1, -1, 4)), matches, lhs_axes=1, rhs_axes=1)
                gt_boxes = F.split(gt_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
                # (B, N, 4)
                gt_boxes = F.stack(*[F.pick(gt_boxes[i], matches, axis=2) for i in range(4)], axis=2)
                # rpn_boxes (B, N, 4)
                # gt_boxes (B, N, 4)
                reg_targets = self.box_encoder(gt_boxes, rpn_boxes)
                # (B, N) -> (B, N, 4)
                box_mask = F.tile(pos_neg_mask.reshape((0, 0, 1)), reps=(1, 1, 4)) == 1
                reg_targets = F.where(box_mask, reg_targets, F.zeros_like(reg_targets))

            # for bboxes target
                num_pos = self.max_pos
                # (B, num_pos, 4)
                reg_targets = F.slice_axis(reg_targets, axis=1, begin=0, end=num_pos)
                reg_targets = F.stop_gradient(reg_targets)
                box_mask = F.slice_axis(box_mask, axis=1, begin=0, end=num_pos)
                box_mask = F.stop_gradient(box_mask)
                # (B, num_pos) val [-1, 0, C) -1: background
                box_cls_id = F.slice_axis(cls_targets, axis=1, begin=0, end=num_pos) - 1

            # (BN, channels, 1, 1) -> (B, N, channels)
            roi_feat = roi_feat.reshape((self.batch_size, self.num_sample, -1))
            # (B, num_pos, channels)
            roi_feat = F.slice_axis(roi_feat, axis=1, begin=0, end=num_pos)
            # (B*num_pos, channels)
            roi_feat = roi_feat.reshape((self.batch_size * num_pos, -1))
            # (B * num_pos, C*4)
            box_pred = self.reg(roi_feat)
            # (B, num_pos, C, 4)
            box_pred = box_pred.reshape((self.batch_size, num_pos, self.num_classes, 4))
            # (B, num_pos, C, 4) -> (B, num_pos, 4)
            box_pred = F.split(box_pred, axis=-1, num_outputs=4, squeeze_axis=True)
            box_pred = F.stack(*[F.pick(box_pred[i], box_cls_id, axis=2) for i in range(4)], axis=2)

            # raw_rpn_scores (B, rpn_num_samples, 1)
            # rpn_masks (B, rpn_num_samples) 1: pos 0: ignore -1: neg
            # raw_rpn_boxes (B, rpn_num_samples, 4)
            # raw_rpn_box_targets (B, rpn_num_samples, 4)
            # cls_pred (B, num_samples, C+1)
            # cls_targets (B, num_samples) value [-1, C) -1: ignore 0: background
            # box_pred (B, num_pos, 4)
            # reg_targets (B, num_pos, 4)
            # box_mask (B, num_pos, 4)
            return (raw_rpn_scores, rpn_masks, raw_rpn_boxes, raw_rpn_box_targets,
                    cls_pred, cls_targets, box_pred, reg_targets, box_mask)

        all_scores = F.softmax(cls_pred, axis=-1)
        # scores, cls_id (B, N, C)
        scores = F.slice_axis(all_scores, axis=-1, begin=1, end=None)
        _BN = F.zeros_like(F.slice_axis(all_scores, axis=-1, begin=0, end=1))
        cls_id = F.broadcast_add(_BN, F.reshape(F.arange(self.num_classes), shape=(1, 1, -1)))
        box_pred = self.reg(roi_feat)
        # (BN, C*4) -> (B, N, C, 4)
        box_pred = box_pred.reshape((self.batch_size, -1, self.num_classes, 4))

        # rpn_boxes (B, N, 4) -> (1, BN, 4)
        rpn_boxes = rpn_boxes.reshape((1, -1, 4))
        # box_pred (B, N, C, 4) -> (C, B, N, 4) -> (C, BN, 4)
        box_pred =F.transpose(box_pred, axes=(2, 0, 1, 3)).reshape((0, -1, 4))
        # boxes (C, BN, 4) -> (C, B, N, 4)
        boxes = self.box_decoder(box_pred, rpn_boxes).reshape((0, self.batch_size, -1, 4))
        # boxes (C, B, N, 4) -> (B, N, C, 4) -> (B, NC, 4)
        boxes = F.transpose(boxes, axes=(1, 2, 0, 3)).reshape((0, -1, 4))
        boxes = self.box_clip(boxes)

        # scores, cls_id (B, NC, 1)
        scores = scores.reshape((0, -1, 1))
        cls_id = cls_id.reshape((0, -1, 1))

        res = F.concat(*[cls_id, scores, boxes], dim=-1)
        res = F.contrib.box_nms(res, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.0001,
                                id_index=0, score_index=1, coord_start=2, force_suppress=self.force_nms)
        # (B, NC, 6) -> (B, post_nms, 6)
        res = F.slice_axis(res, axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(res, axis=-1, begin=0, end=1)
        scores = F.slice_axis(res, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(res, axis=-1, begin=2, end=6)

        return ids, scores, bboxes

    def _pyramid_roi_feats(self, F, features, rpn_rois, roi_canonical_scale=224.0, eps=1e-6):
        """Assign rpn_rois to specific FPN layers according to its area
           and then perform `ROIPooling` or `ROIAlign` to generate final
           region proposals aggregated features.
        Parameters
        ----------
        features : list of mx.ndarray or mx.symbol [P2, P3, P4, P5[, P6]]
            Features extracted from FPN base network
        rpn_rois : mx.ndarray or mx.symbol
            (N, 5) with [[batch_index, x1, y1, x2, y2], ...] like
        roi_size : tuple
            The size of each roi with regard to ROI-Wise operation
            each region proposal will be roi_size spatial shape.
        strides : tuple e.g. [4, 8, 16, 32]
            Define the gap that roi image and feature map have
        roi_mode : str, default is align
            ROI pooling mode. Currently support 'pool' and 'align'.
        roi_canonical_scale : float, default is 224.0
            Hyperparameters for the RoI-to-FPN level mapping heuristic.
        Returns
        -------
        Pooled roi features aggregated according to its roi_level
        """
        num_layers = len(features)
        if num_layers > 4:  # do not use p6 for RCNN
            num_layers = 4
        min_level = 2
        max_level = min_level + num_layers - 1
        _, x1, y1, x2, y2 = F.split(rpn_rois, axis=-1, num_outputs=5)
        h = y2 - y1
        w = x2 - x1
        roi_level = F.floor(4 + F.log2(F.sqrt(w * h) / roi_canonical_scale + eps))
        roi_level = F.squeeze(F.clip(roi_level, min_level, max_level))

        pooled_roi_feats = []
        for i, l in enumerate(range(2, 5 + 1)):
            if self.roi_mode == 'pool':
                # Pool features with all rois first, and then set invalid pooled features to zero,
                # at last ele-wise add together to aggregate all features.
                pooled_feature = F.ROIPooling(features[i], rpn_rois, self.roi_size, 1. / self.strides[i])
                pooled_feature = F.where(roi_level == l, pooled_feature, F.zeros_like(pooled_feature))
            elif self.roi_mode == 'align':
                if 'box_encode' in F.contrib.__dict__ and 'box_decode' in F.contrib.__dict__:
                    # TODO(jerryzcn): clean this up for once mx 1.6 is released.
                    masked_rpn_rois = F.where(roi_level == l, rpn_rois, F.ones_like(rpn_rois) * -1.)
                    pooled_feature = F.contrib.ROIAlign(features[i], masked_rpn_rois, self.roi_size,
                                                        1. / self.strides[i], sample_ratio=2)
                else:
                    pooled_feature = F.contrib.ROIAlign(features[i], rpn_rois, self.roi_size,
                                                        1. / self.strides[i], sample_ratio=2)
                    pooled_feature = F.where(roi_level == l, pooled_feature, F.zeros_like(pooled_feature))
            else:
                raise ValueError("Invalid roi mode: {}".format(self.roi_mode))
            pooled_roi_feats.append(pooled_feature)
        # Ele-wise add to aggregate all pooled features
        pooled_roi_feats = F.ElementWiseSum(*pooled_roi_feats)
        # Sort all pooled features by asceding order
        # [2,2,..,3,3,...,4,4,...,5,5,...]
        # pooled_roi_feats = F.take(pooled_roi_feats, roi_level_sorted_args)
        # pooled roi feats (B*N, C, 7, 7), N = N2 + N3 + N4 + N5 = num_roi, C=256 in ori paper
        return pooled_roi_feats

