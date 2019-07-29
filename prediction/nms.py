import torch


def nms(input_list, shape):
    '''
    remove the region proposals with large overlaps
    :param input: region proposal results. shape batch, max_rp, 5
    :return:
    '''

    zero = torch.tensor([0.]).cuda()
    zeros = torch.tensor([0.] * 5).cuda()

    SMOOTH = torch.tensor([1e-6]).cuda()
    hh, ww = shape

    inputs, scores = input_list
    # remove low score pred bbox
    scores = scores.unsqueeze(2).repeat(1, 1, 5)
    inputs = torch.where(scores > 0.5, inputs, zeros)

    outputs = inputs.clone()
    outputs[:, :, 0] = inputs[:, :, 0] * ww
    outputs[:, :, 1] = inputs[:, :, 1] * hh
    outputs[:, :, 2] = inputs[:, :, 2] * ww
    outputs[:, :, 3] = inputs[:, :, 3] * hh

    area = torch.max((outputs[:, :, 2] - outputs[:, :, 0] + 1) * (outputs[:, :, 3] - outputs[:, :, 1] + 1),
                     torch.tensor([0.]).cuda())

    # remove invalid bbox
    c = outputs[:, :, 0] < outputs[:, :, 2]
    c = c.unsqueeze(2).repeat(1, 1, 5)
    outputs = torch.where(c, outputs, zeros)

    c = outputs[:, :, 1] < outputs[:, :, 3]
    c = c.unsqueeze(2).repeat(1, 1, 5)
    outputs = torch.where(c, outputs, zeros)

    for batch_id, input in enumerate(outputs):
        # outputs[batch_id, :] = torchvision.ops.nms(input[:, 4], torch.rand(input[:, 4].shape).cuda(), 0.6)

        x1 = input[:, 0]
        y1 = input[:, 1]
        x2 = input[:, 2]
        y2 = input[:, 3]
        idxs = torch.argsort(y2)
        pick = torch.zeros(idxs.shape).byte()
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            i = idxs[len(idxs) - 1]
            pick[i] = 1

            xx1 = torch.max(x1[i], x1[idxs])
            yy1 = torch.max(y1[i], y1[idxs])
            xx2 = torch.min(x2[i], x2[idxs])
            yy2 = torch.min(y2[i], y2[idxs])

            # compute the width and height of the RPs
            # w = torch.max(xx2 - xx1, zero)
            # h = torch.max(yy2 - yy1, zero)
            w = torch.max(xx1 - xx2 + 1, zero)
            h = torch.max(yy1 - yy2 + 1, zero)

            # compute the ratio of overlap
            overlap = (w * h) / (area[batch_id, idxs] + SMOOTH)

            # keep all indexes from the index list that have small overlaps
            idxs = idxs[overlap < 0.5]
            idxs = idxs[:-1]

        pick = pick.reshape(1, -1).repeat(5, 1).transpose(0, 1)
        outputs[batch_id, :] = torch.where(pick.cuda(), input, zeros)

    return outputs
