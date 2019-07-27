import torch

SMOOTH = 1e-6

'''
this loss function handles both bbox prediction 
and computing complexity loss

# bbox format, (x1, y1, x2, y2)
'''


def iou_loss(bboxes_pred: torch.Tensor, bboxes_label: torch.Tensor, max_bbox_num=32):

    iou_losses = []
    complexity_losses = []

    for bbox_id in range(bboxes_label.shape[1]):

        bbox_label = bboxes_label[:, bbox_id, :4]
        complexity_label = bboxes_label[:, bbox_id, -1]

        bbox_label = bbox_label.reshape(bbox_label.shape[0], 1, bbox_label.shape[1])
        bbox_label = bbox_label.repeat(1, max_bbox_num, 1)

        #complexity_label = complexity_label.reshape(complexity_label.shape[0], 1, 1)
        #complexity_label = complexity_label.repeat(1, max_bbox_num, 1)

        x1 = torch.max(bboxes_pred[:, :, 0], bbox_label[:, :, 0])
        y1 = torch.max(bboxes_pred[:, :, 1], bbox_label[:, :, 1])
        x2 = torch.min(bboxes_pred[:, :, 2], bbox_label[:, :, 2])
        y2 = torch.min(bboxes_pred[:, :, 3], bbox_label[:, :, 3])

        # calculate intersection
        inter_area = (x2 - x1) * (y2 - y1)

        # intersection area cannot be neg, clean invalid ones
        inter_area = torch.where(x2 > x1, inter_area, torch.tensor([0.]).cuda())
        inter_area = torch.where(y2 > y1, inter_area, torch.tensor([0.]).cuda())

        bbox_pred_area = (bboxes_pred[:, :, 2] - bboxes_pred[:, :, 0]) * (
                bboxes_pred[:, :, 3] - bboxes_pred[:, :, 1])

        # bbox area cannot be neg, clean invalid ones
        bbox_pred_area = torch.where(bboxes_pred[:, :, 2] > bboxes_pred[:, :, 0], bbox_pred_area,
                                     torch.tensor([0.]).cuda())
        bbox_pred_area = torch.where(bboxes_pred[:, :, 3] > bboxes_pred[:, :, 1], bbox_pred_area,
                                     torch.tensor([0.]).cuda())

        bbox_label_area = (bbox_label[:, :, 2] - bbox_label[:, :, 0]) * (
                bbox_label[:, :, 3] - bbox_label[:, :, 1])

        # calculate union & iou
        union = bbox_pred_area + bbox_label_area - inter_area

        iou = inter_area.float() / (SMOOTH + union.float())

        # reset iou for those padding bbox
        iou = torch.where((bbox_label[:, :, 0] + bbox_label[:, :, 1] + bbox_label[:, :, 2] + bbox_label[:, :, 3]) == 0,
                          torch.tensor([1.]).cuda(), iou)

        # find the best matched bbox in the prediction
        iou, indices = torch.max(iou, 1)
        iou_losses.append((1 - iou).mean())

        # complexity_pred shape: (batch, max bbox num, 1)
        # find the complexity diff of the best matched bbox.
        complexity_pred = bboxes_pred[:, :, 4]
        complexity_pred = complexity_pred[torch.arange(complexity_pred.shape[0]), indices]
        complexity_loss = torch.abs(complexity_pred - complexity_label)
        complexity_losses.append(complexity_loss.mean())

    return sum(iou_losses) / max_bbox_num , sum(complexity_losses) / max_bbox_num
