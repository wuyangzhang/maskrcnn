import torch
import torch.nn.functional as F

SMOOTH = 1e-6

'''
which bounding box will be the correct label for the prediction???

O(n2) to search all of them. 

100, 1, 4 vs 100, 30, 4 => 100, 4 
for loop
'''


def iou_loss(bboxes_pred: torch.Tensor, bboxes_label: torch.Tensor):
    # bbox format, (x1, y1, x2, y2)

    iou_loss = []
    for bbox_id in range(bboxes_label.shape[1]):
        bbox_label = bboxes_label[:, bbox_id, :]
        bbox_label = bbox_label.reshape(bbox_label.shape[0], 1, bbox_label.shape[1])
        bbox_label = bbox_label.repeat(1, 30, 1)

        x1 = torch.max(bboxes_pred[:, :, 0], bbox_label[:, :, 0])
        y1 = torch.max(bboxes_pred[:, :, 1], bbox_label[:, :, 1])
        x2 = torch.min(bboxes_pred[:, :, 2], bbox_label[:, :, 2])
        y2 = torch.min(bboxes_pred[:, :, 3], bbox_label[:, :, 3])

        inter_area = (x2 - x1) * (y2 - y1)

        # intersection area cannot be neg, clean invalid ones
        inter_area = torch.where(x2 > x1, inter_area, torch.tensor([0]).int().cuda())
        inter_area = torch.where(y2 > y1, inter_area, torch.tensor([0]).int().cuda())

        bbox_pred_area = (bboxes_pred[:, :, 2] - bboxes_pred[:, :, 0]) * (
                    bboxes_pred[:, :, 3] - bboxes_pred[:, :, 1])

        # bbox area cannot be neg, clean invalid ones
        bbox_pred_area = torch.where(bboxes_pred[:, :, 2] > bboxes_pred[:, :, 0], bbox_pred_area,
                                     torch.tensor([0]).int().cuda())
        bbox_pred_area = torch.where(bboxes_pred[:, :, 3] > bboxes_pred[:, :, 1], bbox_pred_area,
                                     torch.tensor([0]).int().cuda())

        bbox_label_area = (bbox_label[:, :, 2] - bbox_label[:, :, 0]) * (
                    bbox_label[:, :, 3] - bbox_label[:, :, 1])
        union = bbox_pred_area + bbox_label_area - inter_area

        iou = inter_area.float() / (SMOOTH + union.float())

        # clean invalid bbox
        # iou = torch.where(bboxes_pred[:, :, 2] < bboxes_pred[:, :, 0], iou, torch.tensor([0.]).cuda())
        # iou = torch.where(bboxes_pred[:, :, 3] < bboxes_pred[:, :, 1], iou, torch.tensor([0.]).cuda())
        iou_loss.append((1 - iou).mean())

    #return sum(iou_loss)
    return (1-iou).mean()
