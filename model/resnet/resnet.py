import mxnet
from mxnet import nd, cpu
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision


class ResNet50(nn.HybridBlock):
    """
    ResNet50

    the options of output_layers
    2: c1
    4: c2
    5: c3
    6: c4
    7: c5
    """

    def __init__(self, output_layers=7, pretrained=False, ctx=cpu(), **kwargs):
        super(ResNet50, self).__init__(**kwargs)
        if isinstance(output_layers, int):
            output_layers = [output_layers,]
        self.output_layers = output_layers
        self.net = vision.resnet50_v1(pretrained=pretrained, ctx=ctx).features

    def hybrid_forward(self, F, x, *args, **kwargs):
        """

        :param F:
        :param x:
        :param args:
        :param kwargs:
        :return: tuple, each item is feature map
        """
        features = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in self.output_layers:
                features.append(x)

        return tuple(features)
