from collections import OrderedDict

import numpy as np
import mxnet
from mxnet import nd, gpu, autograd, cpu, gluon, contrib, init
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision

from utils.bbox import bbox_decode, clip_bbox, bbox_encode


def generate_anchors(feature_map_shape, image_shape, scales, ratios):
    def center_points(length, step):
        points = nd.arange(0, length)
        return (points + 0.5) * step

    hs = center_points(feature_map_shape[0], image_shape[0] / feature_map_shape[0])
    hs = nd.dot(hs.expand_dims(axis=1), nd.ones((1, feature_map_shape[1])))

    ws = center_points(feature_map_shape[1], image_shape[1] / feature_map_shape[1])
    ws = nd.dot(nd.ones((feature_map_shape[0], 1)), ws.expand_dims(axis=0))

    grid = nd.stack(ws, hs, ws, hs, axis=2)

    whs = []
    for i, scale in enumerate(scales):
        _ratios = ratios[i]
        w = scale * nd.sqrt(nd.array(_ratios))
        h = scale / nd.sqrt(nd.array(_ratios))
        whs.append(nd.stack(w, h, axis=1))
    whs = nd.concat(*whs, dim=0)

    whs = nd.concat(-whs / 2, whs / 2, dim=1)

    # anchors: (H, W, num_anchors_per_position_per_feature_map, 4)
    anchors = grid.reshape(feature_map_shape[0], feature_map_shape[1], 1, 4) + whs.reshape(1, -1, 4)
    anchors = clip_bbox(anchors, image_shape)
    return anchors


def split_batch(batch, num_batch):
    """

    :param batch: OrderDict
    :return:
    """
    _list = [OrderedDict() for _ in range(num_batch)]
    for k, v in batch.items():
        for i in range(num_batch):
            _list[i][k] = v[i]
    return _list


def _combine_keys(data):
    values = []
    for v in data.values():
        shape = v.shape
        H, W, num_anchors = shape[0:3]
        new_shape = (H*W*num_anchors,) + shape[3:]
        values.append(v.reshape(new_shape))
    values = nd.concat(*values, dim=0)
    return values


def match_target(iou_matrix, threshold):
    iou_matrix = iou_matrix.asnumpy()
    gt_max_id = np.argmax(iou_matrix, axis=0)
    max_mask = np.eye(iou_matrix.shape[0])[gt_max_id].T.astype(np.bool)
    iou_matrix[max_mask] = 1

    matched_indices = np.argmax(iou_matrix, axis=1)
    matched_max = np.max(iou_matrix, axis=1)
    bg_mask = matched_max < threshold
    matched_indices[bg_mask] = -1
    return matched_indices


class RegionProposalNetwork(nn.Block):
    def __init__(self, scales, ratios, fg_threshold, bg_threshold, batch_size_per_image, positive_fraction,
                 pre_nms_top_n_in_train, post_nms_top_n_in_train, pre_nms_top_n_in_test, post_nms_top_n_in_test,
                 nms_thresh, **kwargs):
        super(RegionProposalNetwork, self).__init__(**kwargs)
        self.scales = scales
        self.ratios = ratios
        self.max_num_anchors_per_position = 0
        for k, _scales in scales.items():
            num_anchors_per_position = 0
            _ratios = ratios[k]
            for i, scale in enumerate(_scales):
                num_anchors_per_position += len(_ratios[i])
            if num_anchors_per_position > self.max_num_anchors_per_position:
                self.max_num_anchors_per_position = num_anchors_per_position

        with self.name_scope():
            self.head = nn.Conv2D(256, 3, padding=1, activation="relu")
            self.object_cls = nn.Conv2D(self.max_num_anchors_per_position, 1)
            self.object_reg = nn.Conv2D(self.max_num_anchors_per_position * 4, 1)

        self._anchors = dict()
        self.fg_threshold =fg_threshold
        self.bg_threshold = bg_threshold
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_top_n_in_train = pre_nms_top_n_in_train
        self.post_nms_top_n_in_train = post_nms_top_n_in_train
        self.pre_nms_top_n_in_test = pre_nms_top_n_in_test
        self.post_nms_top_n_in_test = post_nms_top_n_in_test
        self.nms_thresh = nms_thresh
        self.object_cls_loss = gloss.SigmoidBCELoss()
        self.object_reg_loss = gloss.HuberLoss()

    def generate_anchors(self, feature_map_shape, image_shape, feature_name):
        anchors = self._anchors.get((feature_map_shape, image_shape, feature_name))
        if anchors is None:
            anchors = generate_anchors(feature_map_shape, image_shape, self.scales[feature_name], self.ratios[feature_name])
            self._anchors[(feature_map_shape, image_shape, feature_name)] = anchors
        return anchors
        # return generate_anchors(feature_map_shape, image_shape, self.scales[feature_name], self.ratios[feature_name])

    def forward(self, features, image_shape, labels=None, bboxes=None):
        """

        :param features: OrderedDict, each features: (B, C, H, W)
        :param image_shape:
        :param labels:
        :param bboxes:
        :return:
        """
        anchors = OrderedDict()  # each anchors:(B, H, W, num_anchors, 4)
        pred_logits = OrderedDict()  # each pred_logits:(B, H, W, num_anchors)
        pred_bbox_deltas = OrderedDict()  # each pred_bbox_deltas:(B, H, W, num_anchors, 4)
        B = 0
        device = cpu()
        for k, feature in features.items():
            if k in self.scales.keys():
                B, C, H, W = feature.shape
                feature = self.head(feature)
                pred_logits[k] = self.object_cls(feature).transpose(axes=(0, 2, 3, 1))
                device = pred_logits[k].context
                pred_bbox_deltas[k] = self.object_reg(feature).transpose(axes=(0, 2, 3, 1)).reshape(B, H, W, -1, 4)
                anchors_per_sample = nd.array(self.generate_anchors(feature.shape[-2:], image_shape, k), ctx=device)
                anchors[k] = nd.stack(*([anchors_per_sample, ] * B), axis=0)

        pred_logits = split_batch(pred_logits, B)
        pred_bbox_deltas = split_batch(pred_bbox_deltas, B)
        anchors = split_batch(anchors, B)

        pred_logits = [_combine_keys(i) for i in pred_logits]              # list each pred_logits:(B*H*W*num_anchors,)
        pred_bbox_deltas = [_combine_keys(i) for i in pred_bbox_deltas]    # list each pred_bbox_deltas:(B*H*W*num_anchors,4)
        anchors = [_combine_keys(i) for i in anchors]                      # list each anchors:(B*H*W*num_anchors, 4)

        object_bboxes_out = []

        rpn_cls_losses = []
        rpn_reg_losses = []

        if not autograd.is_training():
            for pred_logits_per_sample, pred_bbox_deltas_per_sample, anchors_per_sample in zip(pred_logits, pred_bbox_deltas, anchors):
                sorted_indices = nd.argsort(pred_logits_per_sample, axis=0, is_ascend=False)
                sorted_indices = sorted_indices[0:min(len(sorted_indices), self.pre_nms_top_n_in_test)]

                pred_logits_per_sample = pred_logits_per_sample[sorted_indices]
                pred_bbox_deltas_per_sample = pred_bbox_deltas_per_sample[sorted_indices]
                anchors_per_sample = anchors_per_sample[sorted_indices]

                object_bboxes = bbox_decode(pred_bbox_deltas_per_sample, anchors_per_sample)
                object_bboxes = clip_bbox(object_bboxes, image_shape)

                nms_input = nd.concat(
                    nd.arange(len(pred_bbox_deltas_per_sample), ctx=device).reshape(-1, 1),  # index of object bboxes
                    nd.sigmoid(pred_logits_per_sample).reshape(-1, 1),
                    object_bboxes, dim=1)
                outs = nd.contrib.box_nms(nms_input, self.nms_thresh, valid_thresh=0.05)
                outs = outs[:min(len(outs), self.post_nms_top_n_in_test)]

                indices, object_bboxes = outs[:, 0], outs[:, 2:6]
                object_bboxes = nd.contrib.boolean_mask(object_bboxes, indices != -1)
                object_bboxes_out.append(object_bboxes)

            return object_bboxes_out, None

        for pred_logits_per_sample, pred_bbox_deltas_per_sample, anchors_per_sample, labels_per_sample, bboxes_per_sample in zip(pred_logits, pred_bbox_deltas, anchors, labels, bboxes):
            gt_mask = labels_per_sample != -1
            labels_per_sample = nd.contrib.boolean_mask(labels_per_sample, gt_mask)
            bboxes_per_sample = nd.contrib.boolean_mask(bboxes_per_sample, gt_mask)

            sorted_indices = nd.argsort(pred_logits_per_sample, axis=0, is_ascend=False)
            sorted_indices = sorted_indices[0:min(len(sorted_indices), self.pre_nms_top_n_in_train)]

            pred_logits_per_sample = pred_logits_per_sample[sorted_indices]
            pred_bbox_deltas_per_sample = pred_bbox_deltas_per_sample[sorted_indices]
            anchors_per_sample = anchors_per_sample[sorted_indices]

            object_bboxes = bbox_decode(pred_bbox_deltas_per_sample, anchors_per_sample)
            object_bboxes = clip_bbox(object_bboxes, image_shape)
            nms_input = nd.concat(
                nd.arange(len(pred_bbox_deltas_per_sample), ctx=device).reshape(-1, 1),  # index of object bboxes
                nd.sigmoid(pred_logits_per_sample).reshape(-1, 1),
                object_bboxes, dim=1)
            outs = nd.contrib.box_nms(nms_input, self.nms_thresh, valid_thresh=0.05)
            outs = outs[:min(len(outs), self.post_nms_top_n_in_train)]

            indices, object_bboxes = outs[:, 0], outs[:, 2:6]
            valid_mask = indices != -1
            indices = nd.contrib.boolean_mask(indices, valid_mask)
            object_bboxes = nd.contrib.boolean_mask(object_bboxes, valid_mask)
            object_bboxes_out.append(object_bboxes)

            pred_logits_per_sample = pred_logits_per_sample[indices]
            pred_bbox_deltas_per_sample = pred_bbox_deltas_per_sample[indices]
            anchors_per_sample = anchors_per_sample[indices]

            iou_matrix = nd.contrib.box_iou(anchors_per_sample, bboxes_per_sample)
            matched_indices = match_target(iou_matrix, self.fg_threshold)
            object_bboxes_gt = bboxes_per_sample[matched_indices]
            indices = np.arange(len(anchors_per_sample))
            fg_mask = matched_indices >= 0
            fg_indices = indices[fg_mask]
            matched_indices = match_target(iou_matrix, self.bg_threshold)
            bg_mask = matched_indices < 0
            bg_indices = indices[bg_mask]

            num_pos_samples = len(fg_indices)
            batch_size = min(int(num_pos_samples / self.positive_fraction), self.batch_size_per_image)
            num_neg_samples = min(len(bg_indices), batch_size - num_pos_samples)
            bg_indices = bg_indices[:num_neg_samples]

            assert len(fg_indices) > 0
            pred_logits_batch = nd.concat(pred_logits_per_sample[fg_indices], pred_logits_per_sample[bg_indices], dim=0)
            pred_labels_batch = nd.concat(nd.ones(len(fg_indices), ctx=device), nd.zeros(len(bg_indices), ctx=device), dim=0)

            rpn_cls_loss = self.object_cls_loss(pred_logits_batch, pred_labels_batch).sum() / batch_size
            rpn_cls_losses.append(rpn_cls_loss)

            pred_bbox_deltas = pred_bbox_deltas_per_sample[fg_indices]
            bbox_deltas = bbox_encode(object_bboxes_gt[fg_indices], anchors_per_sample[fg_indices])
            rpn_reg_loss = self.object_reg_loss(pred_bbox_deltas, bbox_deltas).sum() / batch_size
            rpn_reg_losses.append(rpn_reg_loss)

        rpn_cls_loss = sum(rpn_cls_losses) / B
        rpn_reg_loss = sum(rpn_reg_losses) / B
        return object_bboxes_out, (rpn_cls_loss, rpn_reg_loss)


class RoIExtractor(nn.Block):
    def __init__(self, feature_map_names, use_fpn=False, **kwargs):
        super(RoIExtractor, self).__init__(**kwargs)
        self.use_fpn = use_fpn
        self.feature_map_names = feature_map_names
        if use_fpn:
            self.levels_map = dict((int(name[-1]), name) for name in self.feature_map_names)
            self.levels_min = min(self.levels_map.keys())
            self.levels_max = max(self.levels_map.keys())
        else:
            assert len(self.feature_map_names) == 1

    def forward(self, features, proposals):
        """

        :param features: OrderedDict, each features: (B, C, H, W)
        :param proposals:
        :return:
        """
        device = features[self.feature_map_names[0]].context
        batch_ids = [nd.full(len(ps), i, ctx=device) for i, ps in enumerate(proposals)]
        B = len(batch_ids)
        batch_ids = nd.concat(*batch_ids, dim=0)
        batch_proposals = nd.concat(*proposals, dim=0)

        if self.use_fpn:
            proposals = nd.concat(batch_ids.reshape(-1, 1), batch_proposals, dim=1)
            ws = batch_proposals[:, 2] - batch_proposals[:, 0]
            hs = batch_proposals[:, 3] - batch_proposals[:, 1]
            areas = ws * hs
            ks = nd.floor(4 + nd.log2(nd.sqrt(areas) / 224))
            ks = nd.clip(ks, self.levels_min, self.levels_max)
            ks = ks.asnumpy()
            batch_indices = np.arange(len(batch_ids))

            _batch_ids = []
            _roi_features = []
            for level, name in self.levels_map.items():
                level_indices = batch_indices[ks == level]
                if len(level_indices) == 0:
                    continue
                level_batch_ids = batch_ids[level_indices]
                roi_features = contrib.ndarray.ROIAlign(features[name], proposals[level_indices], (7, 7), 0.5 ** level)
                _batch_ids.append(level_batch_ids)
                _roi_features.append(roi_features)
            batch_ids = nd.concat(*_batch_ids, dim=0)
            batch_ids = batch_ids.asnumpy()
            roi_features = nd.concat(*_roi_features, dim=0)
            features_split = []
            for i in range(B):
                i_mask = batch_ids == i
                i_indices = batch_indices[i_mask]
                features_split.append(roi_features[i_indices])
            return features_split

        else:
            features = features[self.feature_map_names[0]]
            features = contrib.ndarray.ROIAlign(features, nd.concat(batch_ids.reshape(-1, 1), batch_proposals, dim=1), (7, 7), 0.5**4)
            features_split = []
            idx = 0
            for num_proposals in [len(ps) for ps in proposals]:
                features_split.append(features[idx:idx+num_proposals])
                idx = idx + num_proposals
            return features_split


class Resnet50Backbone(nn.Block):

    def __init__(self, pretrained, ctx=cpu(), **kwargs):
        super(Resnet50Backbone, self).__init__(**kwargs)
        self.output_layers = {
            2: "c1",
            4: "c2",
            5: "c3",
            6: "c4",
            7: "c5",
        }
        self._net = vision.resnet50_v1(pretrained=pretrained, ctx=ctx).features

    def forward(self, X):
        return extract_features(X, self._net, self.output_layers)


class LateralConnection(nn.Block):
    def __init__(self, is_top=False, **kwargs):
        super(LateralConnection, self).__init__(**kwargs)
        self.is_top = is_top
        with self.name_scope():
            self.c_conv = nn.Conv2D(256, 1)
            if not is_top:
                self.p_conv = nn.Conv2D(256, 3, padding=1)
            self.c_bn = nn.BatchNorm()
            if not is_top:
                self.p_bn = nn.BatchNorm()

    def forward(self, low_features, high_features):
        c = self.c_conv(low_features)
        c = self.c_bn(c)
        if self.is_top:
            return c

        p = nd.UpSampling(high_features, scale=2, sample_type="nearest")
        h = min(c.shape[2], p.shape[2])
        w = min(c.shape[3], p.shape[3])
        p = self.p_conv(c[:, :, 0:h, 0:w] + p[:, :, 0:h, 0:w])
        return self.p_bn(p)


class FeaturePyramidNetwork(nn.Block):
    def __init__(self, **kwargs):
        super(FeaturePyramidNetwork, self).__init__(**kwargs)
        self.output_layers = [
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
        ]
        with self.name_scope():
            self.lateral5 = LateralConnection(is_top=True, prefix='lateral5_')
            self.lateral4 = LateralConnection(prefix='lateral4_')
            self.lateral3 = LateralConnection(prefix='lateral3_')
            self.lateral2 = LateralConnection(prefix='lateral2_')
            self.upsample = nn.AvgPool2D()

    def forward(self, features):
        out = OrderedDict(p2=None, p3=None, p4=None, p5=None, p6=None)
        out["p5"] = self.lateral5(features["c5"], None)
        out["p4"] = self.lateral4(features["c4"], out["p5"])
        out["p3"] = self.lateral3(features["c3"], out["p4"])
        out["p2"] = self.lateral2(features["c2"], out["p3"])

        out["p6"] = self.upsample(out["p5"])
        return out


class FasterRCNNDetector(nn.Block):

    def __init__(self, num_classes,
                 # rpn
                 anchor_scales, anchor_ratios, rpn_fg_threshold=0.5, rpn_bg_threshold=0.3, rpn_batch_size_per_image=256,
                 rpn_positive_fraction=0.3,
                 rpn_pre_nms_top_n_in_train=2000, rpn_post_nms_top_n_in_train=1000,
                 rpn_pre_nms_top_n_in_test=2000, rpn_post_nms_top_n_in_test=1000,
                 rpn_nms_thresh=0.7, use_fpn=False,
                 # head
                 fg_threshold=0.5, batch_size_per_image=256, positive_fraction=0.5,
                 max_objs_per_images=100, nms_thresh=0.7,
                 backbone_pretrained=True, ctx=cpu(),
                 **kwargs):
        super(FasterRCNNDetector, self).__init__(**kwargs)
        self.backbone = Resnet50Backbone(backbone_pretrained, ctx)
        self.use_fpn = use_fpn
        if use_fpn:
            self.fpn = FeaturePyramidNetwork(prefix='fpn_')
        self.rpn = RegionProposalNetwork(anchor_scales, anchor_ratios, rpn_fg_threshold, rpn_bg_threshold, rpn_batch_size_per_image,
                                         rpn_positive_fraction, rpn_pre_nms_top_n_in_train, rpn_post_nms_top_n_in_train,
                                         rpn_pre_nms_top_n_in_test, rpn_post_nms_top_n_in_test, rpn_nms_thresh, prefix="rpn_")
        if use_fpn:
            self.roi_extractor = RoIExtractor(self.fpn.output_layers[:-1], use_fpn)
        else:
            self.roi_extractor = RoIExtractor(["c5"])
        self.head = nn.Sequential(prefix="head_")
        with self.head.name_scope():
            self.head.add(nn.Dense(1024, activation='relu'))
            self.head.add(nn.Dense(1024, activation='relu'))
        self.num_classes = num_classes
        self.cls = nn.Dense(num_classes)
        self.reg = nn.Dense(num_classes * 4)
        self.fg_threshold = fg_threshold
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.max_objs_per_images = max_objs_per_images
        self.nms_thresh = nms_thresh
        self.cls_loss = gloss.SoftmaxCrossEntropyLoss()
        self.reg_loss = gloss.HuberLoss()

    def forward(self, images, scale_factors=None, labels=None, bboxes=None):
        B = images.shape[0]
        image_shape = images.shape[-2:]
        device = images.context
        features = self.backbone(images)
        if self.use_fpn:
            features = self.fpn(features)
        proposals, rpn_losses = self.rpn(features, image_shape, labels, bboxes)

        if autograd.is_training():
            _proposals = []
            for proposals_per_sample, labels_per_sample, bboxes_per_sample in zip(proposals, labels, bboxes):
                valid_mask = labels_per_sample != -1
                bboxes_per_sample = nd.contrib.boolean_mask(bboxes_per_sample, valid_mask)
                _proposals_per_sample = nd.concat(bboxes_per_sample, proposals_per_sample, dim=0)
                _proposals.append(_proposals_per_sample)
            proposals = _proposals

        features = self.roi_extractor(features, proposals)

        num_preoposals = [len(ps) for ps in proposals]
        features = nd.concat(*features, dim=0)
        features = self.head(features)
        pred_logits = self.cls(features)
        pred_bbox_deltas = self.reg(features)

        if not autograd.is_training():
            idx = 0
            pred_labels_out = []
            scores_out = []
            pred_bboxes_out = []
            for i, num_preoposals_per_sample in enumerate(num_preoposals):
                pred_logits_per_sample = pred_logits[idx:idx + num_preoposals_per_sample]
                pred_bbox_deltas_per_sample = pred_bbox_deltas[idx:idx+num_preoposals_per_sample]
                anchors = proposals[i]
                idx += num_preoposals_per_sample

                scores = nd.softmax(pred_logits_per_sample, axis=1)
                pred_labels = nd.argmax(scores, axis=1)
                labels_mask = nd.one_hot(pred_labels, self.num_classes)
                pred_bbox_deltas_per_sample = pred_bbox_deltas_per_sample.reshape(-1, self.num_classes, 4).reshape(-1, 4)
                labels_mask = labels_mask.reshape(-1, )
                pred_bbox_deltas_per_sample = nd.contrib.boolean_mask(pred_bbox_deltas_per_sample, labels_mask)
                pred_bboxes = bbox_decode(pred_bbox_deltas_per_sample, anchors)
                scores = nd.max(scores, axis=1)

                nms_input = nd.concat(
                    nd.arange(len(pred_labels), ctx=device).reshape(-1, 1),   # index of bboxes
                    scores.reshape(-1, 1),
                    pred_bboxes,
                    pred_labels.reshape(-1, 1),                   # class id
                    dim=1
                )
                outs = nd.contrib.box_nms(nms_input, self.nms_thresh, 0.05, id_index=6, background_id=0)
                outs = outs[:min(len(outs), self.max_objs_per_images)]
                pred_labels = outs[:, 6]
                scores = outs[:, 1]
                pred_bboxes = outs[:, 2:6]

                if scale_factors is not None:
                    scale_factor = scale_factors[i]
                    if not any(scale_factor == -1):
                        pred_bboxes /= scale_factor

                if len(pred_labels) < self.max_objs_per_images:
                    _pred_labels = nd.full((self.max_objs_per_images,), -1, ctx=device, dtype=pred_labels.dtype)
                    _pred_labels[0:len(pred_labels)] = pred_labels
                    _scores = nd.full((self.max_objs_per_images,), -1, ctx=device, dtype=scores.dtype)
                    _scores[0:len(scores)] = scores
                    _pred_bboxes = nd.full((self.max_objs_per_images, 4), -1, ctx=device, dtype=pred_bboxes.dtype)
                    _pred_bboxes[0:len(pred_bboxes), :] = pred_bboxes

                pred_labels_out.append(pred_labels)
                scores_out.append(scores)
                pred_bboxes_out.append(pred_bboxes)

            pred_labels_out = nd.stack(*pred_labels_out, axis=0)
            scores_out = nd.stack(*scores_out, axis=0)
            pred_bboxes_out = nd.stack(*pred_bboxes_out, axis=0)
            return pred_labels_out, scores_out, pred_bboxes_out
        else:
            cls_losses = []
            reg_losses = []
            idx = 0
            for i, num_preoposals_per_sample in enumerate(num_preoposals):
                pred_logits_per_sample = pred_logits[idx:idx + num_preoposals_per_sample]
                pred_bbox_deltas_per_sample = pred_bbox_deltas[idx:idx+num_preoposals_per_sample]
                anchors = proposals[i]
                labels_per_sample = labels[i]
                bboxes_per_sample = bboxes[i]
                valid_mask = labels_per_sample != -1
                labels_per_sample = nd.contrib.boolean_mask(labels_per_sample, valid_mask)
                bboxes_per_sample = nd.contrib.boolean_mask(bboxes_per_sample, valid_mask)
                idx += num_preoposals_per_sample

                iou_matrix = nd.contrib.box_iou(anchors, bboxes_per_sample)
                matched_indices = match_target(iou_matrix, self.fg_threshold)
                labels_per_sample = labels_per_sample[matched_indices]       # "-1" in matched_incdices will return "0" in output
                bboxes_per_sample = bboxes_per_sample[matched_indices]
                bbox_indices = np.arange(len(matched_indices))
                fg_mask = matched_indices != -1
                bg_mask = matched_indices == -1
                fg_indices = bbox_indices[fg_mask]
                bg_indices = bbox_indices[bg_mask]
                labels_per_sample = nd.array(labels_per_sample, ctx=device)
                labels_per_sample[bg_mask] = 0                               # this line can be ignored

                num_pos_samples = len(fg_indices)
                batch_size = min(int(num_pos_samples / self.positive_fraction), self.batch_size_per_image)
                num_neg_samples = min(len(bg_indices), batch_size - num_pos_samples)
                bg_indices = bg_indices[:num_neg_samples]

                # class loss
                cls_samples_indices = np.concatenate([fg_indices, bg_indices])
                cls_loss = self.cls_loss(
                    pred_logits_per_sample[cls_samples_indices],
                    labels_per_sample[cls_samples_indices]
                ).sum() / batch_size
                cls_losses.append(cls_loss)

                # reg loss
                bbox_deltas_per_sample = bbox_encode(bboxes_per_sample[fg_indices], anchors[fg_indices])
                pred_bbox_deltas_pos = pred_bbox_deltas_per_sample[fg_indices]
                labels_pos = labels_per_sample[fg_indices]
                labels_pos_mask = nd.one_hot(labels_pos, self.num_classes)
                pred_bbox_deltas_pos = pred_bbox_deltas_pos.reshape(-1, 4)
                labels_pos_mask = labels_pos_mask.reshape(-1)
                pred_bbox_deltas_pos = nd.contrib.boolean_mask(pred_bbox_deltas_pos, labels_pos_mask)
                reg_loss = self.reg_loss(pred_bbox_deltas_pos, bbox_deltas_per_sample).sum() / batch_size
                reg_losses.append(reg_loss)
            cls_loss = sum(cls_losses) / B
            reg_loss = sum(reg_losses) / B

            rpn_cls_loss, rpn_reg_loss = rpn_losses
            return rpn_cls_loss, rpn_reg_loss, cls_loss, reg_loss


def extract_features(input, net, output_layers):
    features = OrderedDict()
    for i, layer in enumerate(net):
        input = layer(input)
        if i in output_layers.keys():
            features[output_layers[i]] = input

    return features


def test_rpn(device):
    labels = nd.array([1, 2], ctx=device)
    bboxes = nd.array([[10, 20, 60, 50],
                       [30, 45, 80, 95]], ctx=device)
    scales = {
        "c4": (32, 64),
    }
    ratios = {
        "c4": ((1, 2, 0.5),) * 2,
    }
    net = RegionProposalNetwork(scales, ratios, 0.5, 0.3, 256, 0.5, 2000, 1000, 2000, 1000, 0.7)
    net.initialize(ctx=device)
    features = OrderedDict()
    features["c4"] = nd.random.randn(1, 3, 5, 5, ctx=device)
    object_bboxes, _ = net(features, (100, 100), labels.expand_dims(axis=0), bboxes.expand_dims(axis=0))
    print(len(object_bboxes))
    object_bboxes = object_bboxes[0]
    print(object_bboxes.shape)
    print(object_bboxes)


def test_match_target():
    iou_matrix = nd.random.uniform(high=1, shape=(8, 3))
    print(match_target(iou_matrix, 0.8))
    print(iou_matrix)


def test_fpn():
    device = gpu(0)
    features = OrderedDict({
        "c2": nd.random.randn(1, 256, 200, 334, ctx=device),
        "c3": nd.random.randn(1, 512, 100, 167, ctx=device),
        "c4": nd.random.randn(1, 1024, 50, 84, ctx=device),
        "c5": nd.random.randn(1, 2048, 25, 42, ctx=device),
    })
    fpn = FeaturePyramidNetwork()
    fpn.initialize(init=init.Xavier(), ctx=device)
    features = fpn(features)
    for k, v in features.items():
        print(k, v.shape)


if __name__ == '__main__':
    test_fpn()


# if __name__ == '__main2__':
#     device = gpu()
#     labels = nd.array([1, 2], ctx=device)
#     bboxes = nd.array([[10, 20, 60, 50],
#                        [30, 45, 80, 95]], ctx=device)
#     scales = {
#         "c4": (32, 64),
#     }
#     ratios = {
#         "c4": ((1, 2, 0.5),) * 2,
#     }
#     net = FasterRCNNDetector(10, scales, ratios, 0.5, 0.3, 256, 0.5, 2000, 1000, 2000, 1000, 0.7)
#     net.initialize(ctx=device)
#     X = nd.random.randn(1, 3, 800, 1333, ctx=gpu())
#     rpn_cls_loss, rpn_reg_loss, cls_loss, reg_loss = net(X, None, labels.expand_dims(axis=0), bboxes.expand_dims(axis=0))
#     print(rpn_cls_loss)
#     print(rpn_reg_loss)
#     print(cls_loss)
#     print(reg_loss)
    # test_rpn(device)
    # test_match_target()
