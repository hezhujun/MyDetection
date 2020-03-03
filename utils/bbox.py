from mxnet import nd


def clip_bbox(bboxes, image_shape):
    shape = bboxes.shape
    if len(shape) > 2:
        bboxes[:, :, :, 0] = nd.clip(bboxes[:, :, :, 0], a_min=0, a_max=image_shape[1])                # x1
        bboxes[:, :, :, 1] = nd.clip(bboxes[:, :, :, 1], a_min=0, a_max=image_shape[0])                # y1
        bboxes[:, :, :, 2] = nd.clip(bboxes[:, :, :, 2], a_min=0, a_max=image_shape[1])                # x2
        bboxes[:, :, :, 3] = nd.clip(bboxes[:, :, :, 3], a_min=0, a_max=image_shape[0])                # y2
        return bboxes
    else:
        return nd.stack(
            nd.clip(bboxes[:, 0], a_min=0, a_max=image_shape[1]),  # x1
            nd.clip(bboxes[:, 1], a_min=0, a_max=image_shape[0]),  # y1
            nd.clip(bboxes[:, 2], a_min=0, a_max=image_shape[1]),  # x2
            nd.clip(bboxes[:, 3], a_min=0, a_max=image_shape[0]),  # y2
            axis=1
        )


def bbox_xywh_2_xyxy(bboxs):
    bboxs[2:] = bboxs[:2] + bboxs[2:]
    return bboxs


def bbox_x1y1x2y2_to_x0y0wh(bboxes):
    x0y0 = (bboxes[:, 0:2] + bboxes[:, 2:]) / 2.0
    wh = bboxes[:, 2:] - bboxes[:, :2]
    return nd.concat(x0y0, wh, dim=1)


def bbox_x0y0wh_to_x1y1x2y2(bboxes):
    half = bboxes[:, 2:] / 2.0
    return nd.concat(bboxes[:, 0:2] - half, bboxes[:, 0:2] + half, dim=1)


def bbox_decode(bbox_deltas, anchors,
                mu_x=0, mu_y=0, mu_w=0, mu_h=0,
                sigma_x=0.1, sigma_y=0.1, sigma_w=0.2, sigma_h=0.2):
    bbox = nd.zeros_like(bbox_deltas)
    anchors = bbox_x1y1x2y2_to_x0y0wh(anchors)
    bbox = nd.stack(anchors[:, 2] * (bbox_deltas[:, 0] * sigma_x + mu_x) + anchors[:, 0],
                     anchors[:, 3] * (bbox_deltas[:, 1] * sigma_y + mu_y) + anchors[:, 1],
                     anchors[:, 2] * nd.exp(bbox_deltas[:, 2] * sigma_w + mu_w),
                     anchors[:, 3] * nd.exp(bbox_deltas[:, 3] * sigma_h + mu_h),
                     axis=1)
    return bbox_x0y0wh_to_x1y1x2y2(bbox)


def bbox_encode(bboxes, anchors,
                mu_x=0, mu_y=0, mu_w=0, mu_h=0,
                sigma_x=0.1, sigma_y=0.1, sigma_w=0.2, sigma_h=0.2):
    anchors = bbox_x1y1x2y2_to_x0y0wh(anchors)
    bboxes = bbox_x1y1x2y2_to_x0y0wh(bboxes)
    return nd.stack(
        ((bboxes[:, 0] - anchors[:, 0]) / anchors[:, 2] - mu_x) / sigma_x,
        ((bboxes[:, 1] - anchors[:, 1]) / anchors[:, 3] - mu_y) / sigma_y,
        (nd.log(bboxes[:, 2] / anchors[:, 2]) - mu_w) / sigma_w,
        (nd.log(bboxes[:, 3] / anchors[:, 3]) - mu_h) / sigma_h,
        axis=1,
    )


if __name__ == '__main2__':
    bboxes = nd.array([[10, 20, 60, 50],
                       [30, 45, 80, 95]])
    bboxes_c = bbox_x1y1x2y2_to_x0y0wh(bboxes)
    bboxes_b = bbox_x0y0wh_to_x1y1x2y2(bboxes_c)

    print(bboxes)
    print(bboxes_c)
    print(bboxes_b)
    print((bboxes - bboxes_b).norm() < 1e-6)

    anchors = nd.array([[10, 20, 60, 50],
                        [30, 45, 80, 95]])
    anchors = anchors / 1.1
    deltas = bbox_encode(bboxes, anchors)
    bboxes_b = bbox_decode(deltas, anchors)
    print(deltas)
    print(bboxes_b)
    print((bboxes - bboxes_b).norm() < 1e-6)

