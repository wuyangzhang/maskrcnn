import torch


def nms(inputs: torch.tensor):
    '''
    remove the region proposals with large overlaps
    :param input: region proposal results. shape batch, max_rp, 5
    :return:
    '''

    x1 = inputs[:, :, 0]
    x2 = inputs[:, :, 1]
    y1 = inputs[:, :, 2]
    y2 = inputs[:, :, 3]
    area = torch.max((x2 - x1 + 1) * (y2 - y1 + 1), torch.tensor([0.]).cuda())

    # iterate batch.. will be slow
    outputs = inputs.clone()

    for batch_id, input in enumerate(inputs):

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

            # find the overlap between the selected rp i
            # and all the other region proposals

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = torch.max(x1[i], x1[idxs[:]])
            yy1 = torch.max(y1[i], y1[idxs[:]])
            xx2 = torch.min(x2[i], x2[idxs[:]])
            yy2 = torch.min(y2[i], y2[idxs[:]])

            # compute the width and height of the RPs
            w = torch.max(xx2 - xx1 + 1, torch.tensor([0.]).cuda())
            h = torch.max(yy2 - yy1 + 1, torch.tensor([0.]).cuda())

            # compute the ratio of overlap
            overlap = (w * h) / (area[batch_id, idxs[:]] + 1e-6)

            # delete all indexes from the index list that have
            idxs = idxs[overlap < 0.8]
            idxs = idxs[:-1]

        pick = pick.reshape(1, -1).repeat(5, 1).transpose(0, 1)
        outputs[batch_id, :] = torch.where(pick.cuda(), input, torch.tensor([0.]*5).cuda())

    return outputs
