import time, pickle
import sys

sys.path.append('/home/nvidia/maskrcnn-benchmark')
import glob
from collections import defaultdict

import numpy as np

import logging

log_output_path = './log/par_50.log'
log_path = '_par_50.pkl'

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=log_output_path,
                    level=logging.INFO,
                    )

mylogger = logging.getLogger('maskrcnn')


# load GT
gt_list = list()
gt_masks = dict()
for gt in glob.glob('./davis_gt_*.pkl'):
    gt_name = gt.split('_')[-1].split('.')[0]
    gt_list.append(gt_name)
    with open(gt, 'rb') as f:
        gt_masks[gt_name] = pickle.load(f)

for subj in gt_list:
    gt_mask = gt_masks[subj]
    # get challenge results
    # with open('eval/mask_rcnn/davis'+subj+'_non_par.pkl', 'rb') as f:
    with open('./data/davis' + subj + log_path, 'rb') as f:
        ch_masks = pickle.load(f)

    print(subj)
    gt_mask = np.transpose(gt_mask, (1, 0, 2, 3))

    best_ious = []
    avgs = []
    for frame_id in range(gt_mask.shape[0]):
        print(subj, frame_id)
        ch_mask = ch_masks[frame_id]
        # if ch_mask is None:
        #     best_ious.append(0)
        #     #mylogger.info(best_iou)
        #     continue
        if len(ch_mask.shape) > 1:
            ch_mask = ch_mask.squeeze(1)  # remove 1 channel

        for gt_m in range(gt_mask.shape[1]):
            gt_tar = gt_mask[frame_id][gt_m]
            # find best matched one at challenge.
            best_iou = 0
            for ch_m in range(ch_mask.shape[0]):
                ch_tar = ch_mask[ch_m].numpy()
                intersection = np.logical_and(gt_tar, ch_tar)
                union = np.logical_or(gt_tar, ch_tar)
                iou = np.sum(intersection) / np.sum(union)
                best_iou = max(best_iou, iou)
                print(best_iou)
            best_ious.append(best_iou)

    avg = sum(best_ious) / len(best_ious)
    avgs.append(avg)
    mylogger.info('subj ,{}, avg iou ,{},'.format(subj, avg))
            # mylogger.info(',{},{}'.format(subj, best_iou))

mylogger.info('avg iou,{}'.format(sum(avgs)/len(avgs)))
