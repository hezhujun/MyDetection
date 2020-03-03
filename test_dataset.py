from datasets.coco import TCTDataset
import os


if __name__ == '__main__':
    dataset = TCTDataset("/root/userfolder/datasets/TCT", "train")
    img, label = dataset[0]
    print(img)
    print(label)
