import mxnet
from mxnet import nd, init, autograd
from mxnet.gluon import nn
import numpy as np
from .box import BoxCornerToCenter


class BoxDecoder(nn.HybridBlock):

    def __init__(self, mu=(0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0), convert_anchor=False, **kwargs):
        """

        :param mu:
        :param std:
        :param convert_anchor: boolean, default is False
            whether to convert anchor from corner to center format.
        :param kwargs:
        """
        super(BoxDecoder, self).__init__(**kwargs)
        self.mu = mu
        self.std = std
        self.convert_anchor = convert_anchor
        if convert_anchor:
            with self.name_scope():
                self.corner_to_center = BoxCornerToCenter(is_split=True)

    def hybrid_forward(self, F, x, anchor):
        """

        :param F:
        :param x: (B, N, 4) box offset
        :param anchor: (1, N, 4)
        :return:
        """
        if self.convert_anchor:
            a = self.corner_to_center(anchor)
        else:
            a = F.split(anchor, axis=-1, num_outputs=4)
        b = F.split(x, axis=-1, num_outputs=4)
        ox = F.broadcast_add(F.broadcast_mul((b[0] * self.std[0] + self.mu[0]), a[2]), a[0])
        oy = F.broadcast_add(F.broadcast_mul((b[1] * self.std[1] + self.mu[1]), a[3]), a[1])
        dw = b[2] * self.std[2] + self.mu[2]
        dh = b[3] * self.std[3] + self.mu[3]
        dw = F.exp(dw)
        dh = F.exp(dh)
        w = F.broadcast_mul(dw, a[2])
        h = F.broadcast_mul(dh, a[3])
        ow = w * 0.5
        oh = h * 0.5
        return F.concat(ox-ow, oy-oh, ox+ow, oy+oh, dim=-1)


class BoxEncoder(nn.HybridBlock):

    def __init__(self, mu=(0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0), convert_anchor=False,
                 convert_box=False, **kwargs):
        """

        :param mu:
        :param std:
        :param convert_anchor: boolean, default is False
            whether to convert anchor from corner to center format.
        :param convert_box: boolean, default is False
            whether to convert boxes from corner to center format.
        :param kwargs:
        """
        super(BoxEncoder, self).__init__(**kwargs)
        self.mu = mu
        self.std = std
        self.convert_anchor = convert_anchor
        self.convert_box = convert_box
        with self.name_scope():
            if convert_anchor or convert_box:
                self.corner_to_center = BoxCornerToCenter(is_split=True)

    def hybrid_forward(self, F, x, anchor):
        """

        :param F:
        :param x: (B, N, 4) box
        :param anchor: (B, N, 4)
        :return:
        """
        if self.convert_anchor:
            a = self.corner_to_center(anchor)
        else:
            a = F.split(anchor, axis=-1, num_outputs=4)

        if self.convert_box:
            b = self.corner_to_center(x)
        else:
            b = F.split(x, axis=-1, num_outputs=4)

        x = (((b[0] - a[0]) / a[2]) - self.mu[0]) / self.std[0]
        y = (((b[1] - a[1]) / a[3]) - self.mu[1]) / self.std[1]

        w = (F.log((b[2] / a[2])) - self.mu[2]) / self.std[2]
        h = (F.log((b[3] / a[3])) - self.mu[3]) / self.std[3]

        return F.concat(x, y, w, h, dim=-1)
