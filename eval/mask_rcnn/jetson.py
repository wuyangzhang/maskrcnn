import time, pickle

import sys

sys.path.append('/home/nvidia/maskrcnn-benchmark')

import cv2
from state_monitor import ControlManager
from app import ApplicationManager
from partition import PartitionManager
from prediction import PredictionManager
from config import Config

import logging

log_output_path = './multi_offload.log'

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=log_output_path,
                    level=logging.INFO,
                    )

log = logging.getLogger('maskrcnn')


def main():
    config = Config()
    config.log = log
    paths = []
    with open('eval/mask_rcnn/davis_videos-nvidia.txt', 'r') as f:
        for line in f.readlines():
            paths.append(line.split()[0])
    par_mgr = PartitionManager(config)
    pred_mgr = PredictionManager(config)
    control_mgr = ControlManager(config, pred_mgr, par_mgr)
    app_mgr = ApplicationManager()

    cur_path = None
    masks = []
    #did = ['bear', 'breakdance-flare', 'boxing-fisheye']
    for img_path in paths:
        # skip = False
        # for d in did:
        #     if d not in img_path:
        #         skip = True
        #         break
        # if skip:
        #     continue
        # remove cache if a new video is playing
        print(img_path)

        path = img_path.split('/')[-2]

        if cur_path is not None and path != cur_path:
            with open('eval/mask_rcnn/data/davis{}_par_50.pkl'.format(cur_path), 'wb+') as f:
                pickle.dump(masks, f)
                masks = []

        if cur_path is None or path != cur_path:
            cur_path = path
            pred_mgr.bbox_queue = list()

        img = cv2.imread(img_path)

        if config.frame_width is None:
            config.frame_height, config.frame_width = img.shape[:2]

        # counting the time when receiving the frame
        branch = control_mgr.get_branch_state()

        if branch == 'distribute':
            s = time.time()
            coords, weights = pred_mgr.get_pred_bbox()
            s2 = time.time()
            print('prediction,', s2-s)
            # get computing capability of involved nodes
            resources = control_mgr.report_resources()

            # perform frame partition based on computing capability
            imgs = par_mgr.frame_partition(img, coords, weights, resources, app_mgr)

            for img in imgs:
                print('offload data size', img.shape)
                log.info('offload size,{}'.format(img.shape))
            s = time.time()
            log.info('frame partition,{}'.format(s - s2))

            # offload the partitions to edge servers
            control_mgr.dist_jobs(imgs)

            # merge all the results
            s = time.time()
            bbox = control_mgr.merge_partitions()
            s2 = time.time()
            log.info('merge partition,{}'.format(s2-s))

            bboxgt, _ = app_mgr.run(img.copy())
            composite_gt = app_mgr.rendering(img.copy(), bboxgt)
            # cv2.imshow('src', composite_gt)
            # cv2.waitKey(10)

            if bbox:
                composite = app_mgr.rendering(img, bbox)
                pred_mgr.add_bbox(bbox)
            else:
                composite = img
            s = time.time()
            print('render,', s-s2)

        elif branch == 'cache_refresh':
            bbox, _ = app_mgr.run(img)
            composite = app_mgr.rendering(img, bbox)
            pred_mgr.add_bbox(bbox)
        else:
            raise NotImplementedError
        if bbox is not None:
            masks.append(bbox.get_field('mask'))
        else:
            masks.append(None)
        # collect bbox here.
        # cv2.imshow('dis', composite)
        # cv2.waitKey(10)

        time.sleep(0.03)
    with open('eval/mask_rcnn/data/davis{}_par_50.pkl'.format(cur_path), 'wb+') as f:
        pickle.dump(masks, f)
        a = 1

cv2.destroyAllWindows()

if __name__ == "__main__":
    #main()

    with open('./eval/mask_rcnn/log/par_50.log', 'r') as f:
        vals = []
        for line in f.readlines():
            val = float(line.split(',')[-2])
            vals.append(val)
        print(sum(vals) / len(vals))
