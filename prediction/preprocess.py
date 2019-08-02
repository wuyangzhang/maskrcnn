import torch

SMOOTH = torch.tensor([1e-6]).cuda()
inf = torch.tensor([float('inf')]).cuda()


def reorder(inputs):
    # shape (b, time, features)
    # 16 * 5 * 160
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1, 4)

    for batch_id in range(inputs.shape[0]):
        free_slot_start_index = torch.sum(inputs[batch_id, 0, :, 0] + inputs[batch_id, 0, :, 1] \
                    + inputs[batch_id, 0, : , 2] + inputs[batch_id, 0, :, 3] != 0)

        # 16 * 5 * 32 * 5
        # iterate window. each time we pick two of them
        # and reorder the second bbox list.
        for t in range(1, inputs.shape[1]):
            # will reorder the bbox at t + 1
            # new order for inputs[:, t:, :, :], shape: 32 * 5
            new_order = torch.zeros(inputs.shape[2], inputs.shape[3]).cuda()
            used = torch.zeros(inputs.shape[2]).byte().cuda()
            used = torch.where(inputs[batch_id, t, :, 0] + inputs[batch_id, t, :, 1] +
                               inputs[batch_id, t, :, 2] + inputs[batch_id, t, :, 3] != 0,
                               used,
                               torch.tensor([1]).byte().cuda()
                               )
            # bbox1 is for t , bbox2 is for t + 1
            bbox1, bbox2 = inputs[batch_id, t-1, :, :], inputs[batch_id, t, :, :]

            bbox2_area = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])

            # iterate bbox at t, and the find the best match one at t + 1
            for box_id in range(inputs.shape[2]):
                #bbox_label => 16 * 32 * 5
                bbox_prev = bbox1[box_id, :].unsqueeze(0).repeat(inputs.shape[2], 1)

                bbox_x_ratio = torch.abs(1-((bbox_prev[:, 0] + bbox_prev[:,  2]) / 2) \
                               / ((bbox2[:, 0] + bbox2[:, 2]) / 2 ))

                bbox_y_ratio = torch.abs(1-((bbox_prev[:, 1] + bbox_prev[:, 3]) / 2) \
                               / ((bbox2[:, 1] + bbox2[:, 3]) / 2))

                mask = (bbox_x_ratio < 0.4) & (bbox_y_ratio < 0.5)

                bbox_prev_area = (bbox_prev[:, 2] - bbox_prev[:, 0]) * (bbox_prev[:, 3] - bbox_prev[:, 1])

                bbox_area_ratio = torch.abs(1 - bbox_prev_area / bbox2_area)

                # mask candidate bboxes that are too far from the target
                bbox_area_ratio = torch.where(mask, bbox_area_ratio, inf)

                 # find the best matched bbox in the prediction
                best_area_ratio, indices = torch.min(bbox_area_ratio, 0)

                # set a hard threshold for the area matching.
                if best_area_ratio < 0.2:
                    new_order[box_id, :] = bbox2[indices, :]
                    used[indices] = 1

            # append unused bbox at t + 1 to new_order. start from free slot.
            unused_len = torch.sum(~used)
            if free_slot_start_index + unused_len < 32:
                new_order[free_slot_start_index: free_slot_start_index + unused_len, :] = bbox2[~used]
            # update free slot
            free_slot_start_index += unused_len

            # reset the bbox  at t+1
            inputs[batch_id, t, :, :] = new_order

    return inputs.reshape(inputs.shape[0], inputs.shape[1], -1)