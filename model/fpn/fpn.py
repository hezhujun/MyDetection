import mxnet
from mxnet import nd, cpu, init
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision


class LateralConnection(nn.HybridBlock):
    def __init__(self, is_top=False, **kwargs):
        super(LateralConnection, self).__init__(**kwargs)
        self.is_top = is_top
        with self.name_scope():
            self.c_conv = nn.Conv2D(256, 1, weight_initializer=init.Normal())
            if not is_top:
                self.p_conv = nn.Conv2D(256, 3, padding=1, weight_initializer=init.Normal())
            self.c_bn = nn.BatchNorm()
            if not is_top:
                self.p_bn = nn.BatchNorm()

    def hybrid_forward(self, F, low_features, high_features):
        c = self.c_conv(low_features)
        c = self.c_bn(c)
        if self.is_top:
            return c

        p = F.UpSampling(high_features, scale=2, sample_type="nearest")
        # The shape of p may bigger than that of c
        p = F.slice_like(p, c, axes=(2, 3))
        p = self.p_conv(c + p)
        return self.p_bn(p)


class FPN(nn.HybridBlock):

    def __init__(self, **kwargs):
        super(FPN, self).__init__(**kwargs)
        with self.name_scope():
            self.lateral5 = LateralConnection(is_top=True, prefix='lateral5_')
            self.lateral4 = LateralConnection(prefix='lateral4_')
            self.lateral3 = LateralConnection(prefix='lateral3_')
            self.lateral2 = LateralConnection(prefix='lateral2_')
            self.upsample = nn.AvgPool2D()

    def hybrid_forward(self, F, *x):
        c2, c3, c4, c5 = x
        p5 = self.lateral5(c5, None)
        p4 = self.lateral4(c4, p5)
        p3 = self.lateral3(c3, p4)
        p2 = self.lateral2(c2, p3)

        p6 = self.upsample(p5)
        return p2, p3, p4, p5, p6
