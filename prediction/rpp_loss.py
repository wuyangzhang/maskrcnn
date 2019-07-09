import torch

SMOOTH = 1e-6


def iou_loss(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    # iterate each sample
    # iterate each region proposal in the output if it is not all zero
    # find the minimal IOU pair in the labels

    loss = 0
    for batch_index, batch in enumerate(outputs):
        for rp_index, rp in enumerate(batch):
            if sum(rp) == 0:
                continue
            max_iou = 0

            for pair in labels[batch_index]:


    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

