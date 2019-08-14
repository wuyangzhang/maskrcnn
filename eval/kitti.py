import pickle
import numpy as np
import logging
import torch


def find_iou(a, b, epsilon=1e-5):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def bbox_compare(bbox1, bbox2):
    res = list()
    for gt_box in bbox1.bbox:
        best_iou = 0
        for pred_box in bbox2.bbox:
            iou = find_iou(gt_box, pred_box)
            best_iou = max(best_iou, iou)
        res.append(best_iou)
    return res


def find_mask_iou(target, prediction):
    target = target.numpy()
    prediction = prediction.numpy()
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    return np.sum(intersection) / np.sum(union)


def seg_compare(bbox1, bbox2):
    res = list()
    for gt_mask in bbox1.get_field('mask'):
        best_iou = 0
        for pred_mask in bbox2.get_field('mask'):
            iou = find_mask_iou(gt_mask, pred_mask)
            best_iou = max(best_iou, iou)
        res.append(best_iou)
    return res


if __name__ == "__main__":

    logging.basicConfig(filename='/home/wuyang/maskrcnn-benchmark/trace.log',
                        format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        )

    # logging.Formatter(fmt='%(asctime)s.%(msecs)03d',datefmt='%Y-%m-%d,%H:%M:%S')

    log = logging.getLogger('mobidist')

    f = open('demo/gt/kitti_100.pkl', 'rb')
    gt = pickle.load(f)
    f.close()

    f = open('demo/par_2/kitti_100.pkl', 'rb')
    eval = pickle.load(f)
    bbox_iou = list()
    mask_iou = list()
    cnt = list()
    for img_id in gt:
        if img_id == 11:
            a = 1
        bbox_res = bbox_compare(gt[img_id], eval[img_id])
        mask_res = seg_compare(gt[img_id], eval[img_id])
        bbox_iou += bbox_res
        mask_iou += mask_res
        for item in bbox_res:
            if type(item) == torch.Tensor:
                item = item.item()
            if item > 0.5:
                cnt.append(1)
            else:
                cnt.append(0)

        log.info('id {} bbox res {}, mask res{}'.format(img_id, bbox_res, mask_res))
        log.info(gt[img_id].bbox)
        log.info(eval[img_id].bbox)
    log.info('eval done! bbox_iou {}, mask iou {}'.format(sum(bbox_iou)/len(bbox_iou), sum(mask_iou)/len(mask_iou)))
    f.close()
