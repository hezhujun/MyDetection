from datasets.coco import TCTDataset
from transforms.transforms import *
from mxnet.gluon.data import DataLoader
from mxnet import nd
from visualize.bbox import draw
from PIL import Image

categories = [
    "background",
    "ascus",
    "asch",
    "lsil",
    "hsil",
    "agc",
    "adenocarcinoma",
    "vaginalis",
    "flora/monilia",
    "dysbacteriosis"
]


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

    for i, sample in enumerate(batch):
        num_objs = sample[1].shape[0]
        labels[i, 0:num_objs] = nd.array(sample[1])
        bboxes[i, 0:num_objs, :] = sample[2]
        scale_factors[i, :] = sample[3]

    return images, labels, bboxes, scale_factors


# def collate_fn2(*args):
#     out = []
#     for items in args:
#         shapes = nd.array([item.shape for item in items])
#         max_shape = nd.max(shapes, axis=0)
#         _items = []
#         for item in items:
#             shape = item.shape
#             _item = nd.full(max_shape, -1, ctx=item.context, dtype=item.dtype)
#             if len(shape) == 1:
#                 _item[:shape[0]] = item
#             elif len(shape) == 2:
#                 _item[:shape[0], :shape[1]] = item
#             elif len(shape) == 3:
#                 _item[:shape[0], :shape[1], :shape[2]] = item
#             elif len(shape) == 4:
#                 _item[:shape[0], :shape[1], :shape[2], :shape[3]] = item
#             else:
#                 raise NotImplementedError()
#             _items.append(_item)
#         _items = nd.stack(*_items, axis=0)
#         out.append(_items)
#     return tuple(out)


if __name__ == '__main__':
    root = "/run/media/hezhujun/DATA1/Document/dataset/TCT"
    train_ann_file = "/run/media/hezhujun/DATA1/Document/dataset/TCT/annotations/train.json"
    val_ann_file = "/run/media/hezhujun/DATA1/Document/dataset/TCT/annotations/val.json"
    test_ann_file = "/run/media/hezhujun/DATA1/Document/dataset/TCT/annotations/test.json"
    train_transforms = Compose([
        Resize((1333, 800), True),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        Normalize(mean=(127, 127, 127), std=(255, 255, 255)),
        ToTensor()
    ])
    val_transforms = Compose([
        Resize((1333, 800), True),
        Normalize(mean=(127, 127, 127), std=(255, 255, 255)),
        ToTensor()
    ])
    # train_dataset = CocoDataset(root, "tct_train", train_ann_file, train_transforms)
    val_dataset = TCTDataset(root, "tct_val", val_ann_file, val_transforms)
    val_data_loader = DataLoader(val_dataset, 4, False, last_batch="keep", batchify_fn=collate_fn)
    for images, labels, bboxes, scale_factors in val_data_loader:
        num_image = images.shape[0]
        for i in range(num_image):
            image = images[i]
            image = image.transpose((1, 2, 0))
            image = image * 255 + 127
            image = image.asnumpy().astype(np.uint8)
            _labels = labels[i]
            print(_labels)
            num_objs = (_labels != -1).sum().asscalar()
            _bboxes = bboxes[i]
            print(_bboxes)
            scale_factor = scale_factors[i]
            print(scale_factor)
            image = draw(image, _labels[0:num_objs].asnumpy(), _bboxes[0:num_objs, :].asnumpy(), category_names=categories)
            img = Image.fromarray(image)
            img.show()
            input()
        break
