import mxnet
from mxnet import nd

if __name__ == '__main__':
    bboxes = nd.array([
        [0, 0, 50, 50],
        [0, 0, 40, 40],
        [50, 50, 100, 100],
        [25, 25, 75, 75],
    ])
    labels = nd.array([1, 0, 2, 3])
    scores = nd.array([0.7, 0.9, 0.5, 0.5])
    background_id = labels == 0
    print(labels.shape)
    print(scores.shape)
    print(bboxes.shape)
    nms_input = nd.concat(
        labels.reshape(-1, 1),
        scores.reshape(-1, 1),
        bboxes,
        dim=1
    )
    outs = nd.contrib.box_nms(nms_input, 0.5, 0.1, topk=10, id_index=0, background_id=0)
    labels = outs[:, 0]
    scores = outs[:, 1]
    bboxes = outs[:, 2:6]
    print(labels)
    print(scores)
    print(bboxes)