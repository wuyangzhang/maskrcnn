import torch



def iou_loss(bboxes_pred: torch.Tensor, bboxes_label: torch.Tensor, max_bbox_num=32):
    '''
    Calculate the iou between each label bbox and all other predicted bboxes.
    Find the predictive bbox with the largest iou as the best matched one.
    :param bboxes_pred:
    :param bboxes_label:
    :param max_bbox_num:
    :return:
    '''
    SMOOTH = torch.tensor([1e-6]).cuda()

    iou_losses = []
    complexity_losses = []
    padding_cnts = []

    zero = torch.tensor([0.]).cuda()
    one = torch.tensor([1.]).cuda()

    # iterate each GT bbox
    for bbox_id in range(bboxes_label.shape[1]):

        # no. bbox_id bbox at each batch sample
        bbox_label = bboxes_label[:, bbox_id, :4]
        non_padding_cnt = torch.sum(bbox_label[:, 0] + bbox_label[:, 1] + bbox_label[:, 2] + bbox_label[:, 3] != 0)

        complexity_label = bboxes_label[:, bbox_id, -1]

        bbox_label = bbox_label.reshape(bbox_label.shape[0], 1, bbox_label.shape[1])
        bbox_label = bbox_label.repeat(1, max_bbox_num, 1)

        # complexity_label = complexity_label.reshape(complexity_label.shape[0], 1, 1)
        # complexity_label = complexity_label.repeat(1, max_bbox_num, 1)

        x1 = torch.max(bboxes_pred[:, :, 0], bbox_label[:, :, 0])
        y1 = torch.max(bboxes_pred[:, :, 1], bbox_label[:, :, 1])
        x2 = torch.min(bboxes_pred[:, :, 2], bbox_label[:, :, 2])
        y2 = torch.min(bboxes_pred[:, :, 3], bbox_label[:, :, 3])

        # calculate intersection between each label bbox and all other predicted bbox.
        inter_area = (x2 - x1) * (y2 - y1)

        # intersection area cannot be neg, clean invalid ones
        inter_area = torch.where(x2 > x1, inter_area, zero)
        inter_area = torch.where(y2 > y1, inter_area, zero)

        bbox_pred_area = (bboxes_pred[:, :, 2] - bboxes_pred[:, :, 0]) * \
                         (bboxes_pred[:, :, 3] - bboxes_pred[:, :, 1])
        bbox_label_area = (bbox_label[:, :, 2] - bbox_label[:, :, 0]) * \
                          (bbox_label[:, :, 3] - bbox_label[:, :, 1])

        # calculate union & iou
        union = bbox_pred_area + bbox_label_area - inter_area

        #iou = inter_area.float() / (SMOOTH + union.float())
        iou = inter_area / (SMOOTH + union)

        # reset iou for those padding bbox as 1
        iou = torch.where(
            (bbox_label[:, :, 0] + bbox_label[:, :, 1] + bbox_label[:, :, 2] + bbox_label[:, :, 3]) == 0,
            one,
            iou)

        # find the best matched bbox in the prediction
        iou, indices = torch.max(iou, 1)

        padding_cnts.append(non_padding_cnt)
        iou_losses.append(torch.sum(1-iou))
        complexity_pred = bboxes_pred[:, :, 4]
        complexity_pred = complexity_pred[torch.arange(complexity_pred.shape[0]), indices]
        complexity_loss = torch.abs(complexity_pred - complexity_label)
        complexity_losses.append(complexity_loss.mean())

    return torch.sum(torch.stack(iou_losses)) / torch.sum(torch.stack(padding_cnts)), sum(complexity_losses) / max_bbox_num
