import torch

from log import mylogger

layers = [3, 4, 6, 3]


def cal_complexity(units):
    '''
    calculate the percentage of how much computing complexity can be saved by SACT.
    :param units: including unit of 4 Resnet blocks
    :return:
    '''

    for block_id, block in enumerate(units):
        # block [0]: ponder cost, block[1] : used layers
        tmp = block[1] / layers[block_id]
        tmp = torch.min(tmp.float(), torch.tensor([1.]).cuda())
        tmp = tmp.mean().tolist()
        mylogger.warn('format:block[x] shows y complexity,{},{}'.format(block_id, tmp))
