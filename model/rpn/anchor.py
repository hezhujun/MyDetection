import mxnet
from mxnet import nd
from mxnet.gluon import nn
import numpy as np


class AnchorGenerator(nn.HybridBlock):

    def __init__(self, stride, scales, ratios, alloc_size, **kwargs):
        super(AnchorGenerator, self).__init__(**kwargs)
        anchors = _generate(stride, scales, ratios, alloc_size)
        self.anchors = self.params.get_constant("anchor_", anchors)

    def hybrid_forward(self, F, x, anchors):
        # anchors (B=1, C=1, H, W, N*4)
        anchors = F.slice_like(anchors, x, axes=(2, 3))
        anchors = F.squeeze(anchors, axis=(0, 1))  # anchors (H, W, N*4)
        return anchors.reshape((0, 0, -1, 4))  # anchors (H, W, N, 4)


def _generate(stride, scales, ratios, alloc_size):
    scales = np.array(scales).reshape(-1, 1)
    ratios = np.array(ratios)
    w = np.sqrt(scales * ratios)
    h = np.sqrt(scales / ratios)
    w = w.reshape(-1, 1)
    h = h.reshape(-1, 1)
    wh = np.hstack([-w, -h, w, h])

    center = stride / 2
    center = np.array([center, center, center, center]).reshape(-1, 4)
    base_anchors = center + wh / 2

    x = np.arange(alloc_size[0]) * stride
    y = np.arange(alloc_size[1]) * stride
    x, y = np.meshgrid(x, y)
    offset_xy = np.stack([x.ravel(), y.ravel(), x.ravel(), y.ravel()], axis=1)

    anchors = offset_xy.reshape((-1, 1, 4)) + base_anchors  # anchors (HW, N, 4)
    # the num dims is no more than 6 to better debug
    anchors = anchors.reshape((1, 1, alloc_size[0], alloc_size[1], -1))  # anchors (B=1, C=1, H, W, N*4)
    return anchors.astype(np.float32)
