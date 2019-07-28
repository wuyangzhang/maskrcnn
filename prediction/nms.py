import torch


def nms(inputs: torch.tensor, shape):
    '''
    remove the region proposals with large overlaps
    :param input: region proposal results. shape batch, max_rp, 5
    :return:
    '''

    hh, ww = shape

    # iterate batch.. will be slow
    outputs = inputs.clone()

    outputs[:, :, 0] = inputs[:, :, 0] * ww
    outputs[:, :, 1] = inputs[:, :, 1] * hh
    outputs[:, :, 2] = inputs[:, :, 2] * ww
    outputs[:, :, 3] = inputs[:, :, 3] * hh

    area = torch.max((inputs[:, :, 2] - inputs[:, :, 0] + 1) * (inputs[:, :, 3] - inputs[:, :, 1] + 1),
                     torch.tensor([0.]).cuda())

    zero = torch.tensor([0.]).cuda()

    a = 1
    idxs = torch.argsort(outputs[:, :, 3])
    pick = torch.zeros(idxs.shape).byte()

    # for batch_id, input in enumerate(outputs):
    #
    #     x1 = input[:, 0]
    #     y1 = input[:, 1]
    #     x2 = input[:, 2]
    #     y2 = input[:, 3]
    #
    #     idxs = torch.argsort(y2)
    #
    #     pick = torch.zeros(idxs.shape).byte()
    #     while len(idxs) > 0:
    #         # grab the last index in the indexes list and add the
    #         # index value to the list of picked indexes
    #         i = idxs[len(idxs) - 1]
    #         pick[i] = 1
    #
    #         # find the largest (x, y) coordinates for the start of
    #         # the bounding box and the smallest (x, y) coordinates
    #         # for the end of the bounding box
    #         xx1 = torch.max(x1[i], x1[idxs[:]])
    #         yy1 = torch.max(y1[i], y1[idxs[:]])
    #         xx2 = torch.min(x2[i], x2[idxs[:]])
    #         yy2 = torch.min(y2[i], y2[idxs[:]])
    #
    #         # compute the width and height of the RPs
    #         w = torch.max(xx2 - xx1 + 1, zero)
    #         h = torch.max(yy2 - yy1 + 1, zero)
    #
    #         # compute the ratio of overlap
    #         overlap = (w * h) / (area[batch_id, idxs[:]] + 1e-6)
    #
    #         # delete all indexes from the index list that have
    #         idxs = idxs[overlap < 0.8]
    #         idxs = idxs[:-1]
    #
    #     pick = pick.reshape(1, -1).repeat(5, 1).transpose(0, 1)
    #     outputs[batch_id, :] = torch.where(pick.cuda(), input, torch.tensor([0.]*5).cuda())
    #
    # return outputs
