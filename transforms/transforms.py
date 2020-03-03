import random

import cv2
import numpy as np
from mxnet import nd


class Resize(object):
    def __init__(self, img_size, keep_ratio=True):
        """

        :param img_size: (w, h)
        :param keep_ratio:
        """
        assert isinstance(img_size, tuple)
        self.img_size = img_size
        self.keep_ratio = keep_ratio

    def __call__(self, image, labels, bboxes, scale_factor, *args):
        origin_shape = image.shape
        origin_size = (origin_shape[1], origin_shape[0])

        if self.keep_ratio:
            ratio = origin_size[0] / origin_size[1]
            if ratio <= self.img_size[0] / self.img_size[1]:
                image = cv2.resize(image, (int(round(self.img_size[1] * ratio)), self.img_size[1]), dst=image)
                assert image.shape[1] <= self.img_size[0]
                assert image.shape[0] == self.img_size[1]
            else:
                image = cv2.resize(image, (self.img_size[0], int(round(self.img_size[0] / ratio))), dst=image)
                assert image.shape[1] == self.img_size[0]
                assert image.shape[0] <= self.img_size[1]
        else:
            image = cv2.resize(image, self.img_size, dst=image)

        img_size = (image.shape[1], image.shape[0])
        fx = (img_size[0] - 1) / (origin_size[0] - 1)
        fy = (img_size[1] - 1) / (origin_size[1] - 1)

        scale_factor = np.array([fx, fy, fx, fy])

        bboxes *= scale_factor

        if self.keep_ratio:
            _img = np.zeros((self.img_size[1], self.img_size[0], 3), image.dtype)
            img_shape = image.shape
            _img[0:img_shape[0], 0:img_shape[1], :] = image
            image = _img

        return (image, labels, bboxes, scale_factor) + tuple(args)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, labels, bboxes, scale_factor, *args):
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std
        return (image, labels, bboxes, scale_factor) + tuple(args)


class ToTensor(object):
    def __call__(self, image, labels, bboxes, scale_factor, *args):
        image = image.transpose((2, 0, 1))
        image = nd.array(image)
        labels = nd.array(labels)
        bboxes = nd.array(bboxes)
        scale_factor = nd.array(scale_factor)
        return (image, labels, bboxes, scale_factor) + tuple(args)


class RandomHorizontalFlip(object):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, image, labels, bboxes, scale_factor, *args):
        if random.random() < self.flip_ratio:
            cv2.flip(image, 1, image)

            w = image.shape[1]
            x1 = w - 1 - bboxes[:, 0]
            y1 = bboxes[:, 1]
            x2 = w - 1 - bboxes[:, 2]
            y2 = bboxes[:, 3]

            bboxes = np.stack([x2, y1, x1, y2], axis=1)
            # the bboxes can not be mapped to original image
            scale_factor = np.array([-1, -1, -1, -1])
        return (image, labels, bboxes, scale_factor) + tuple(args)


class RandomVerticalFlip(object):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, image, labels, bboxes, scale_factor, *args):
        if random.random() < self.flip_ratio:
            cv2.flip(image, 0, image)

            h = image.shape[0]
            x1 = bboxes[:, 0]
            y1 = h - 1 - bboxes[:, 1]
            x2 = bboxes[:, 2]
            y2 = h - 1 - bboxes[:, 3]

            bboxes = np.stack([x1, y2, x2, y1], axis=1)
            # the bboxes can not be mapped to original image
            scale_factor = np.array([-1, -1, -1, -1])

        return (image, labels, bboxes, scale_factor) + tuple(args)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
