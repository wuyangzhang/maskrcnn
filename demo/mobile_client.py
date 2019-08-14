import time
import threading
import pickle
import matplotlib.pyplot as plt

import cv2

from state_monitor import ControlManager
from partition import PartitionManager
from prediction import PredictionManager
from config import Config
from app import ApplicationManager
from demo.remote_server import Server
from dataset import MobiDistDataset



def main():



    # init the main components of MobiDist
    config = Config()
    dataset = MobiDistDataset(config).getDataLoader()

    app_mgr = ApplicationManager()

    # launch remote servers, only for the debug purpose
    for i in config.servers:
        print('running remote server {}'.format(i))
        p = threading.Thread(target=Server, args=(config.servers[i], 1024 * 1024), daemon=True)
        p.start()

    print('wait seconds for launching remote servers..')
    time.sleep(2)

    par_mgr = PartitionManager(config)
    pred_mgr = PredictionManager(config)
    control_mgr = ControlManager(config, pred_mgr, par_mgr)

    e2e_latency = []

    res = dict()
    for id, img in enumerate(dataset):

        if id < 201:
            continue

        img = img[0].numpy()
        if config.frame_width is None:
            config.frame_height, config.frame_width = img.shape[:2]

        # counting the time when receiving the frame.
        start_time = time.time()

        composite = None

        src = img.copy()
        app_mgr.run(src, display=True)

        branch = control_mgr.get_branch_state()

        if branch == 'distribute':

            # for debug
            src = img.copy()
            bbox, _ = app_mgr.run(src)

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

            if bbox:
                composite = app_mgr.rendering(img, bbox)
                pred_mgr.add_bbox(bbox)
            else:
                composite = img

        elif branch == 'cache_refresh':
            bbox, _ = app_mgr.run(img)
            composite = app_mgr.rendering(img, bbox)
            pred_mgr.add_bbox(bbox)

        elif control_mgr.branch_states[branch] == 'shortcut':
            composite = None

        b, g, r = cv2.split(composite)
        composite = cv2.merge([r, g, b])
        plt.imshow(composite)
        plt.title('distributed{}'.format(id))
        plt.show()

        res[id] = bbox
        if id >= 100 and id % 100 == 0:
            f = open("demo/par_2/kitti_{}.pkl".format(id), "wb")
            pickle.dump(res, f)
            f.close()
            res = dict()

        latency = time.time() - start_time
        print("Time: {:.2f} s / img".format(latency))
        e2e_latency.append(latency)

        if cv2.waitKey(1) == 27:
            break  # esc to quit


cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
