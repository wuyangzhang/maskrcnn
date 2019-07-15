import cv2
import time

from frame_filter import Filter
from state_monitor import ControlManager
from partition import PartitionManager
from prediction import PredictionManager
from config import Config
from demo import MaskCompute
from dataset import MobiDistDataset


def main():
    cam = cv2.VideoCapture(0)
    mask_engine = MaskCompute()
    filter_actor = Filter()

    # init the main components of MobiDist
    config = Config()
    dataset = MobiDistDataset(config).getDataLoader()
    par_mgr = PartitionManager(config)
    pred_mgr = PredictionManager(config)
    control_mgr = ControlManager(config, pred_mgr, par_mgr, mask_engine)

    e2e_latency = []
    # while True:
    for img in dataset:
        img = img[0].numpy()
        # counting the time when receiving the frame.
        start_time = time.time()
        #_, img = cam.read() # shape (480, 640, 3)

        img = filter_actor.filter(img)
        composite = None

        if img is not None:
            branch = control_mgr.get_branch_state()
            if branch == 'distribute':

                bbox_coord, bbox_weight = pred_mgr.get_pred_bbox()

                # get computing capability of involved nodes
                resources = control_mgr.report_resources()

                # perform frame partition based on computing capability
                # and computing overheads of jos
                imgs = par_mgr.frame_partition(img, bbox_coord, bbox_weight, resources)

                # offload the partitions to edge servers
                control_mgr.dist_jobs(imgs)

                # merge all the results
                composite, bbox_coord, overhead = control_mgr.merge_partitions()

                pred_mgr.add_bbox(bbox_coord, overhead)

            elif branch == 'cache_refresh':
                composite, bbox_coord, overhead = mask_engine.run(img)
                print(bbox_coord)
                pred_mgr.add_bbox(bbox_coord, overhead)

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
