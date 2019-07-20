import time
import threading

import cv2

from frame_filter import Filter
from state_monitor import ControlManager
from partition import PartitionManager
from prediction import PredictionManager
from config import Config
from app import ApplicationManager
from demo.remote_server import Server
from dataset import MobiDistDataset


def main():
    # cam = cv2.VideoCapture(0)
    app_mgr = ApplicationManager()
    filter_actor = Filter()

    # init the main components of MobiDist
    config = Config()

    # launch remote servers, only for the debug purpose
    for i in config.servers:
        print('running remote server {}'.format(i))
        p = threading.Thread(target=Server, args=(config.servers[i], 1024*1024), daemon=True)
        p.start()

    time.sleep(3)
    print('wait for launching servers..')
    dataset = MobiDistDataset(config).getDataLoader()
    par_mgr = PartitionManager(config)
    pred_mgr = PredictionManager(config)
    control_mgr = ControlManager(config, pred_mgr, par_mgr, app_mgr)

    e2e_latency = []
    # while True:
    for img in dataset:
        img = img[0].numpy()
        if config.frame_width is None:
            config.frame_height, config.frame_width = img.shape[:2]

        # counting the time when receiving the frame.
        start_time = time.time()
        # _, img = cam.read() # shape (480, 640, 3)

        img = filter_actor.filter(img)
        composite = None

        if img is not None:
            branch = control_mgr.get_branch_state()
            if branch == 'distribute':

                coords, weights = pred_mgr.get_pred_bbox()

                # get computing capability of involved nodes
                resources = control_mgr.report_resources()

                # perform frame partition based on computing capability
                # and computing overheads of jos
                imgs = par_mgr.frame_partition(img, coords, weights, resources)

                # offload the partitions to edge servers
                control_mgr.dist_jobs(imgs)

                # merge all the results
                bbox = control_mgr.merge_partitions()

                pred_mgr.add_bbox(bbox)

                print('######we have {} box in total#######'.format(len(bbox.bbox)))

                composite = app_mgr.rendering(img, bbox)

            elif branch == 'cache_refresh':
                composite, bbox = app_mgr.run(img)

                pred_mgr.add_bbox(bbox)

            elif control_mgr.branch_states[branch] == 'shortcut':
                composite = None

            cv2.imshow("COCO detections", composite)
            cv2.waitKey(100)
        else:
            cv2.imshow("COCO detections", control_mgr.last_composite)

        latency = time.time() - start_time
        print("Time: {:.2f} s / img".format(latency))
        e2e_latency.append(latency)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
