import numpy as np
from mxnet import nd
from mxnet.gluon import utils as gutils


def inference(detector, data_loader, device):
    if isinstance(device, list):
        is_multi_gpus = True
    else:
        is_multi_gpus = False

    results = []
    for images, labels, bboxes, scale_factors, image_ids in data_loader:
        print("inference image", [i for i in image_ids.asnumpy()])
        if is_multi_gpus:
            images = gutils.split_and_load(images, device)
            labels = gutils.split_and_load(labels, device)
            bboxes = gutils.split_and_load(bboxes, device)
            scale_factors = gutils.split_and_load(scale_factors, device)
            image_ids = gutils.split_and_load(image_ids, device)

            pred_labels_batch, pred_scores_batch, pred_bboxes_batch = [], [], []

            for images_, labels_, bboxes_, scale_factors_, image_ids_ in zip(images, labels, bboxes, scale_factors, image_ids):
                pred_labels, pred_scores, pred_bboxes = detector(images_, scale_factors_)
                pred_labels_batch.append(pred_labels)
                pred_scores_batch.append(pred_scores)
                pred_bboxes_batch.append(pred_bboxes)

            image_ids_batch = image_ids
            for pred_labels, pred_scores, pred_bboxes, image_ids in zip(pred_labels_batch, pred_scores_batch, pred_bboxes_batch, image_ids_batch):
                pred_labels = pred_labels.asnumpy()
                pred_scores = pred_scores.asnumpy()
                pred_bboxes = pred_bboxes.asnumpy()
                image_ids = image_ids.asnumpy()

                for pred_labels_per_image, pred_scores_per_image, pred_bboxes_per_image, image_id in zip(pred_labels, pred_scores, pred_bboxes, image_ids):
                    valid_mask = pred_labels_per_image != -1
                    indices = np.arange(len(valid_mask))[valid_mask]
                    for index in indices:
                        results.append({
                            "image_id": int(image_id),
                            "category_id": int(pred_labels_per_image[index]),
                            "bbox": [float(i) for i in pred_bboxes_per_image[index]],
                            "score": float(pred_scores_per_image[index]),
                        })

                nd.waitall()

        else:
            images = images.as_in_context(device)
            labels = labels.as_in_context(device)
            bboxes = bboxes.as_in_context(device)
            scale_factors = scale_factors.as_in_context(device)

            pred_labels, pred_scores, pred_bboxes = detector(images, scale_factors)

            pred_labels = pred_labels.asnumpy()
            pred_scores = pred_scores.asnumpy()
            pred_bboxes = pred_bboxes.asnumpy()
            image_ids = image_ids.asnumpy()

            for pred_labels_per_image, pred_scores_per_image, pred_bboxes_per_image, image_id in zip(pred_labels, pred_scores, pred_bboxes, image_ids):
                valid_mask = pred_labels_per_image != -1
                indices = np.arange(len(valid_mask))[valid_mask]
                for index in indices:
                    results.append({
                        "image_id": int(image_id),
                        "category_id": int(pred_labels_per_image[index]),
                        "bbox": [float(i) for i in pred_bboxes_per_image[index]],
                        "score": float(pred_scores_per_image[index]),
                    })

    return results
