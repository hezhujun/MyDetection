import mxnet
from mxnet import nd, init, autograd
from mxnet.gluon import nn
import numpy as np


class BoxCornerToCenter(nn.HybridBlock):

    def __init__(self, is_split=False, **kwargs):
        super(BoxCornerToCenter, self).__init__(**kwargs)
        self.is_split = is_split

    def hybrid_forward(self, F, x):
        x1, y1, x2, y2 = F.split(x, axis=-1, num_outputs=4)
        w = x2 - x1
        h = y2 - y1
        x = x1 + w * 0.5
        y = y1 + h * 0.5
        if self.is_split:
            return x, y, w, h
        else:
            return F.concat(x, y, w, h, dim=-1)


class BoxClip(nn.HybridBlock):

    def __init__(self, x_max, y_max, **kwargs):
        super(BoxClip, self).__init__(**kwargs)
        self.x_max = x_max
        self.y_max = y_max

    def hybrid_forward(self, F, x):
        x1, y1, x2, y2 = F.split(x, axis=-1, num_outputs=4)
        x1 = F.clip(x1, a_min=0, a_max=self.x_max)
        y1 = F.clip(y1, a_min=0, a_max=self.y_max)
        x2 = F.clip(x2, a_min=0, a_max=self.x_max)
        y2 = F.clip(y2, a_min=0, a_max=self.y_max)
        return F.concat(x1, y1, x2, y2, dim=-1)
