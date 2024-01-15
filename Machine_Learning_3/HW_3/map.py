import torch
import torchvision
import numpy as np

def batch_average_precision(batch_pred_boxes, batch_scores, batch_true_boxes, threshold):
    number_of_predictions = sum(len(boxes) for boxes in batch_pred_boxes)
    batch_is_active = []

    for pred_boxes, pred_scores, true_boxes in zip(batch_pred_boxes, batch_scores, batch_true_boxes):
        ious = torchvision.ops.box_iou(pred_boxes, true_boxes)
        max_iou_value, max_iou_arg = ious.max(dim=0)

        active_index = max_iou_arg[max_iou_value > threshold]
        inactive_index = max_iou_arg[max_iou_value <= threshold]

        detected = torch.zeros(len(pred_boxes))
        detected[active_index] = 1
        detected[inactive_index] = 0

        batch_is_active.append(detected)

    batch_scores = torch.cat(batch_scores)
    batch_is_active = torch.cat(batch_is_active)

    score_sort = torch.argsort(batch_scores)
    batch_scores[score_sort]
    batch_is_active[score_sort]

    f_presision = torch.cumsum(batch_is_active, dim=0) / torch.arange(1, len(batch_is_active) + 1)
    f_recall = torch.cumsum(batch_is_active, dim=0) / (number_of_predictions - 1)

    b_presision = torch.flip(f_presision, dims=(0,))
    b_recall = torch.flip(f_recall, dims=(0,))

    b_y, _ = torch.cummax(b_presision, 0)
    b_x = b_recall

    f_y = torch.flip(b_y, dims=(0,))
    f_x = torch.flip(b_x, dims=(0,))

    dx = f_x[1:] - f_x[:-1]

    a_p = torch.sum(f_y[:-1] * dx)

    return a_p.item()


def mean_average_precision(batch_pred_boxes, batch_scores, batch_true_boxes, batch_true_labels, threshold):
    
    classes = torch.unique(torch.cat(batch_true_labels))
    aps = []

    for i_class in classes:
        ids_true = [labels == i_class for labels in batch_true_labels]
        ids_pred = [labels.argmax(axis=1) == i_class for labels in batch_scores]

        batch_pred_boxes_class = [pb[ids] for ids, pb in zip(ids_pred, batch_pred_boxes) if ids.sum() > 0]
        batch_scores_class = [s[ids].max(axis=1)[0] for ids, s in zip(ids_pred, batch_scores) if ids.sum() > 0]
        batch_true_boxes_class = [tb[ids] for ids, tb in zip(ids_true, batch_true_boxes) if ids.sum() > 0]

        if len(batch_scores_class) == 0:
            continue

        aps.append(batch_average_precision(batch_pred_boxes_class, batch_scores_class, batch_true_boxes_class, threshold))


    return np.mean(aps)