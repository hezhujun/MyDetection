import cv2
import numpy as np

color_table = {
    0: (0x00, 0x00, 0x00),
    1: (0xFF, 0x00, 0x00),
    2: (0xFF, 0x99, 0x00),
    3: (0xFF, 0x00, 0xFF),
    4: (0x00, 0x00, 0xFF),
    5: (0x00, 0x66, 0x00),
    6: (0x66, 0x66, 0x00),
    7: (0x66, 0x33, 0x66),
    8: (0x00, 0x99, 0xCC),
    9: (0x99, 0x00, 0x99),
    10: (0x00, 0x33, 0x66),
    11: (0x33, 0x66, 0x33),
    12: (0x99, 0x99, 0x33),
    13: (0x99, 0x33, 0x99),
    14: (0xCC, 0xCC, 0x33),
    15: (0xD2, 0x69, 0x1E),
    16: (0x6A, 0x5A, 0xCD),
    17: (0x00, 0x80, 0x80),
    18: (0xD2, 0xB4, 0x8C),
}


def draw_dotted_rectangle(img, pt1, pt2, color, thickness, interval=10):
    width_points = list(np.arange(pt1[0], pt2[0], interval))
    if width_points[-1] != pt2[0]:
        width_points.append(pt2[0])

    height_points = list(np.arange(pt1[1], pt2[1], interval))
    if height_points[-1] != pt2[1]:
        height_points.append(pt2[1])

    for i in range(0, len(width_points), 2):
        if i + 1 >= len(width_points):
            break
        cv2.line(img, (width_points[i], pt1[1]), (width_points[i + 1], pt1[1]), color, thickness)
        cv2.line(img, (width_points[i], pt2[1]), (width_points[i + 1], pt2[1]), color, thickness)

    for i in range(0, len(height_points), 2):
        if i + 1 >= len(height_points):
            break
        cv2.line(img, (pt1[0], height_points[i]), (pt1[0], height_points[i + 1]), color, thickness)
        cv2.line(img, (pt2[0], height_points[i]), (pt2[0], height_points[i + 1]), color, thickness)


def draw_text(image, text, point, color):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    margin = 8
    text_area_point2 = (point[0] + text_size[0] + 2 * margin, point[1] + text_size[1] + 2 * margin)
    cv2.rectangle(image, point, text_area_point2, color, -1)
    text_point1 = (point[0] + margin, point[1] + margin + text_size[1])
    cv2.putText(image, text, text_point1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw(image, labels=None, bboxes=None, pred_labels=None, scores=None, pred_bboxes=None, category_names=None):
    if category_names is not None:
        assert isinstance(category_names, (list, tuple))

    if labels is not None and bboxes is not None:
        for label, bbox in zip(labels, bboxes):
            color = color_table[label]
            bbox = np.array(bbox).astype(np.int64)  # cast to int
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=1)
            if category_names is not None and label < len(category_names):
                label_str = category_names[label]
            else:
                label_str = ""
            draw_text(image, label_str, (bbox[0], bbox[1]), color)

    if pred_labels is not None and scores is not None and pred_bboxes is not None:
        for pred_label, score, pred_bbox in zip(pred_labels, scores, pred_bboxes):
            color = color_table[pred_label]
            pred_bbox = np.array(pred_bbox).astype(np.int64)
            draw_dotted_rectangle(image, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), color, thickness=1)

            if category_names is not None and pred_label < len(category_names):
                label_str = category_names[pred_label]
            else:
                label_str = ""
            label_str = "{} {:.3f}".format(label_str, score)

            draw_text(image, label_str, (pred_bbox[0], pred_bbox[1]), color)

    return image
