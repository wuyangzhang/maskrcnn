import time

import sys

sys.path.append('/home/nvidia/maskrcnn-benchmark')

import cv2

from state_monitor import ControlManager
from app import ApplicationManager
from partition import PartitionManager
from prediction import PredictionManager
from config import Config
from dataset import MobiDistDataset


def main():
    config = Config()
    dataset = MobiDistDataset(config).getDataLoader()
    par_mgr = PartitionManager(config)
    pred_mgr = PredictionManager(config)
    control_mgr = ControlManager(config, pred_mgr, par_mgr)
    app_mgr = ApplicationManager()

    for id, img in enumerate(dataset):

        img = img[0].numpy()
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
            imgs = par_mgr.frame_partition(img, coords, weights, resources)
            s = time.time()
            print('frame partition,', s - s2)

            # offload the partitions to edge servers
            control_mgr.dist_jobs(imgs)

            # merge all the results
            s = time.time()
            bbox = control_mgr.merge_partitions()
            s2 = time.time()
            print('merge partition,', s2-s)

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

        elif control_mgr.branch_states[branch] == 'shortcut':
            composite = None

        if cv2.waitKey(1) == 27:
            break  # esc to quit

cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
