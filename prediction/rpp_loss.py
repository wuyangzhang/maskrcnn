import torch
import torch.nn as nn


def iou_loss(bboxes_pred_in, bboxes_label: torch.Tensor, max_bbox_num=32):
    '''
    Calculate the iou between each label bbox and all other predicted bboxes.
    Find the predictive bbox with the largest iou as the best matched one.
    :param bboxes_pred:
    :param bboxes_label:
    :param max_bbox_num:
    :return:
    '''

    bboxes_pred, scores = bboxes_pred_in
    SMOOTH = torch.tensor([1e-6]).cuda()

    iou_losses = []
    complexity_losses = []
    nonpadding_cnts = []

    zero = torch.tensor([0.]).cuda()
    one = torch.tensor([1.]).cuda()
    mse_loss = nn.MSELoss()

    # iterate each GT bbox
    for bbox_id in range(bboxes_label.shape[1]):

        # no. bbox_id bbox at each batch sample
        bbox_label = bboxes_label[:, bbox_id, :4]
        non_padding_cnt = torch.sum(bbox_label[:, 0] + bbox_label[:, 1] + bbox_label[:, 2] + bbox_label[:, 3] != 0)

        complexity_label = bboxes_label[:, bbox_id, -1]

        bbox_label = bbox_label.reshape(bbox_label.shape[0], 1, -1)
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

        nonpadding_cnts.append(non_padding_cnt)
        iou_losses.append(torch.sum(1-iou))
        complexity_pred = bboxes_pred[:, :, 4]
        complexity_pred = complexity_pred[torch.arange(complexity_pred.shape[0]), indices]
        complexity_loss = mse_loss(complexity_pred, complexity_label)
        complexity_losses.append(complexity_loss)

    loss_iou = torch.sum(torch.stack(iou_losses)) / (torch.sum(torch.stack(nonpadding_cnts)) + SMOOTH)

    # reverse way bbox score calculate
    '''
    If the score is larger than a threshold, and it is not a padding, the network will consider this 
    prediction as a valid one. In this case, we will calculate its iou loss.
    '''
    score_losses = list()
    valid_cnts = list()
    for bbox_id in range(bboxes_pred.shape[1]):

        # get each pred bbox (32, 4)
        bbox_pred = bboxes_pred[:, bbox_id, :4]

        # check it the bbox is a non padding one? (32)
        non_padding = (bbox_pred[:, 0] + bbox_pred[:, 1] + bbox_pred[:, 2] + bbox_pred[:, 3] != 0)
        # check the score value (32)
        score = scores[:, bbox_id]
        high_scores = score > 0.4
        # a bbox is a valid one if it is not padding and comes with a high score
        valid = non_padding & high_scores
        valid_cnt = torch.sum(valid)

        score = score.reshape(score.shape[0], 1)
        score = score.repeat(1, max_bbox_num)

        # (batch, max_bbox num, 4)
        bbox_pred = bbox_pred.reshape(bbox_pred.shape[0], 1, -1)
        bbox_pred = bbox_pred.repeat(1, max_bbox_num, 1)

        # bbox_pred[:, :, 0] (32, 32); bboxes_label[:, :, 0] (32, 32)
        x1 = torch.max(bbox_pred[:, :, 0], bboxes_label[:, :, 0])
        y1 = torch.max(bbox_pred[:, :, 1], bboxes_label[:, :, 1])
        x2 = torch.min(bbox_pred[:, :, 2], bboxes_label[:, :, 2])
        y2 = torch.min(bbox_pred[:, :, 3], bboxes_label[:, :, 3])

        # inter_area (32, 32)
        inter_area = (x2 - x1) * (y2 - y1)
        inter_area = torch.where(x2 > x1, inter_area, zero)
        inter_area = torch.where(y2 > y1, inter_area, zero)

        bbox_pred_area = (bbox_pred[:, :, 2] - bbox_pred[:, :, 0]) * \
                         (bbox_pred[:, :, 3] - bbox_pred[:, :, 1])
        bbox_label_area = (bboxes_label[:, :, 2] - bboxes_label[:, :, 0]) * \
                          (bboxes_label[:, :, 3] - bboxes_label[:, :, 1])

        # union (32, 32)
        union = bbox_pred_area + bbox_label_area - inter_area

        # iou (32, 32)
        iou = inter_area / (SMOOTH + union)

        # set iou as zero if the bbox is padding
        iou = torch.where(
            (bbox_pred[:, :, 0] + bbox_pred[:, :, 1] + bbox_pred[:, :, 2] + bbox_pred[:, :, 3] == 0),
            one,
            iou
        )

        iou = torch.where(
            score <= 0.4,
            one,
            iou
        )

        # iou (32)
        iou, indices = torch.max(iou, 1)
        valid_cnts.append(valid_cnt)
        score_losses.append(torch.sum(1-iou))

    loss_score = torch.sum(torch.stack(score_losses)) \
                 / (torch.sum(torch.stack(valid_cnts)) + SMOOTH)
    return loss_iou, loss_score, sum(complexity_losses) / max_bbox_num
