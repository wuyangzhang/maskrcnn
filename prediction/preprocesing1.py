import torch

SMOOTH = torch.tensor([1e-6]).cuda()
inf = torch.tensor([float('inf')]).cuda()
zero = torch.tensor([0.]).cuda()


def remove_tiny_bbox(inputs):

    area = (inputs[:, :, 2] - inputs[:, :, 0]) * (inputs[:, :, 3] - inputs[:, :, 1])
    # find & remove tiny objects..
    is_tiny_area = (area <= 0.0015).unsqueeze(-1).repeat(1, 1, 5)
    return torch.where(is_tiny_area, zero, inputs)


def reorder(inputs):
    '''
    Two steps flow.
    construct a new_order tensor. (batch, 32, 4)
    1) find the best matching bbox at t+1 for each bbox at t.
    2) append those failing to match bbox (new objects appearing at t) to
    :param inputs:
    :return:
    '''

    #inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1, 4)

    area = (inputs[:, :, :, 2] - inputs[:, :, :, 0]) * (inputs[:, :, :, 3] - inputs[:, :, :, 1])

    # find & remove tiny objects..
    is_tiny_area = (area <= 0.0015).unsqueeze(-1).repeat(1, 1, 1, 4)
    inputs = torch.where(is_tiny_area, torch.tensor([0.]).cuda(), inputs)

    area = (inputs[:, :, :, 2] - inputs[:, :, :, 0]) * (inputs[:, :, :, 3] - inputs[:, :, :, 1])

    free_slot_start_index = torch.sum(inputs[:, 0, :, 0] + inputs[:, 0, :, 1] \
                                      + inputs[:, 0, :, 2] + inputs[:, 0, :, 3] != 0, dim=1)

    # 16 * 5 * 32 * 5
    for t in range(1, inputs.shape[1]):

        # whether a bbox at t+1 has been selected as the best matching for bbox at t
        # used = torch.zeros(inputs.shape[0], inputs.shape[2]).byte().cuda()
        # used = torch.where(inputs[:, t, :, 0] + inputs[:, t, :, 1] +
        #                    inputs[:, t, :, 2] + inputs[:, t, :, 3] != 0,
        #                    used,
        #                    torch.tensor([1]).byte().cuda()
        #                    )

        # will reorder the bbox at t + 1
        # new order for inputs[:, t:, :, :], shape: batch * 32 * 5
        new_order = torch.zeros(inputs.shape[0], inputs.shape[2], inputs.shape[3]).cuda()

        # bbox1 is for t , bbox2 is for t + 1
        bbox1, bbox2 = inputs[:, t - 1, :, :], inputs[:, t, :, :]

        #bbox2_area = (bbox2[:, :, 2] - bbox2[:, :, 0]) * (bbox2[:, :, 3] - bbox2[:, :, 1])
        bbox2_area = area[:, t, :]
        # iterate bbox at t, and the find the best match one at t + 1
        for box_id in range(inputs.shape[2]):
            # bbox_label => 16 * 32 * 5
            bbox_prev = bbox1[:, box_id, :].unsqueeze(1).repeat(1, inputs.shape[2], 1)

            # calculate x offset.
            bbox_x_ratio = torch.abs(1 - ((bbox_prev[:, :, 0] + bbox_prev[:, :, 2]) / 2) \
                                     / ((bbox2[:, :, 0] + bbox2[:, :, 2]) / 2))

            # calculate y offset.
            bbox_y_ratio = torch.abs(1 - ((bbox_prev[:, :, 1] + bbox_prev[:, :, 3]) / 2) \
                                     / ((bbox2[:, :, 1] + bbox2[:, :, 3]) / 2))

            offset_mask = (bbox_x_ratio < 0.4) & (bbox_y_ratio < 0.5)

            bbox_prev_area = (bbox_prev[:, :, 2] - bbox_prev[:, :, 0]) * (bbox_prev[:, :, 3] - bbox_prev[:, :, 1])

            bbox_area_ratio = torch.abs(1 - bbox_prev_area / bbox2_area)

            # mask candidate box that are too far from the target
            bbox_area_ratio = torch.where(offset_mask, bbox_area_ratio, torch.tensor([float('inf')]).cuda())

            # find the best matched bbox in the prediction, shape batch, 32
            #best_area_ratio, indices = torch.min(bbox_area_ratio, 1)

            best_ratio, indices = torch.min(bbox_area_ratio + bbox_x_ratio + bbox_y_ratio, 1)

            # set a hard threshold for the area matching. (batch)
            area_ratio_mask = best_ratio < 0.4

            new_order[area_ratio_mask, box_id, :] = bbox2[area_ratio_mask, indices[area_ratio_mask], :]

            # reuse previous bbox as the value
            new_order[~area_ratio_mask, box_id, :] = bbox1[~area_ratio_mask, box_id, :]

            #used[area_ratio_mask, indices[area_ratio_mask]] = 1

        # for batch_id in range(used.shape[0]):
        #     # append unused bbox at t + 1 to new_order. start from free slot.
        #     unused_len = torch.sum(~used[batch_id], dim=0)
        #     if free_slot_start_index[batch_id] + unused_len < 32:
        #         new_order[batch_id, free_slot_start_index[batch_id]: free_slot_start_index[batch_id] + unused_len, :] = bbox2[batch_id, ~used[batch_id]]
        #         # update free slot
        #     free_slot_start_index[batch_id] += unused_len

            # reset the bbox  at t+1
        inputs[:, t, :, :] = new_order

    return inputs.reshape(inputs.shape[0], inputs.shape[1], -1)


if __name__ == "__main__":
    from prediction import RPPNDataset
    from prediction.lstm import LSTM
    from config import Config

    config = Config()
    model = LSTM(input_size=160, hidden_size=64, window=config.window_size, num_layers=2).cuda()

    test_video_files = config.home_addr + 'kitty/testing/seq_list.txt'
    dataset = 'kitti'
    eval_data = RPPNDataset(test_video_files, dataset)
    eval_data_loader = eval_data.getDataLoader(batch_size=32, window_size=config.window_size, shuffle=True)

    for batch_id, data in enumerate(eval_data_loader):
        train_x, train_y, path = data
        # train_x = train_x.cuda()
        train_x = reorder(train_x.cuda())
