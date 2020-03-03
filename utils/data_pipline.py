import numpy as np
from mxnet import nd


class Collator():

    def __init__(self, max_objs):
        self.max_objs = max_objs

    def __call__(self, batch):
        num_batch = len(batch)
        b = [items for items in zip(*batch)]
        image = nd.stack(*b[0])

        labels = nd.full((num_batch, self.max_objs), -1)
        bboxes = nd.full((num_batch, self.max_objs, 4), -1)
        for i, (_labels, _bboxes) in enumerate(zip(b[1], b[2])):
            num_objs = _labels.shape[0]
            labels[i, :num_objs] = _labels
            bboxes[i, :num_objs] = _bboxes

        scale_factor = nd.stack(*b[3])
        other = b[4:]
        return (image, labels, bboxes, scale_factor) + tuple(other)


def collate_fn(batch):
    images = nd.stack(*[sample[0] for sample in batch], axis=0)

    num_samples = len(batch)
    max_num_objs = batch[0][1].shape[0]
    for sample in batch:
        if sample[1].shape[0] > max_num_objs:
            max_num_objs = sample[1].shape[0]

    labels = nd.full((num_samples, max_num_objs), -1, dtype=np.int64)  # -1 means no objects
    bboxes = nd.full((num_samples, max_num_objs, 4), -1, dtype=np.float32)
    # -1 means bbox can not be mapped to to original image
    scale_factors = nd.full((num_samples, 4), -1, dtype=np.float32)
    image_ids = nd.full((num_samples, ), -1, dtype=np.int64)

    for i, sample in enumerate(batch):
        num_objs = sample[1].shape[0]
        labels[i, 0:num_objs] = nd.array(sample[1])
        bboxes[i, 0:num_objs, :] = sample[2]
        scale_factors[i, :] = sample[3]
        image_ids[i] = sample[4]

    return images, labels, bboxes, scale_factors, image_ids
