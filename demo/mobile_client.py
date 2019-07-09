import argparse
import cv2
import time

from maskrcnn_benchmark.config import cfg
from frame_filter import Filter
from state_monitor import ControlManager
from partition import PartitionManager
from prediction import PredictionManager
from demo.predictor import COCODemo
from . import pondercost_proc


class MaskCompute:

    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
        parser.add_argument(
            "--config-file",
            # default="configs/e2e_mask_rcnn_R_50_FPN_1x.yaml",
            default="./configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.7,
            help="Minimum score for the prediction to be shown",
        )
        parser.add_argument(
            "--min-image-size",
            type=int,
            default=224,
            help="Smallest size of the image to feed to the model. "
                 "Model was trained with 800, which gives best results",
        )
        parser.add_argument(
            "--show-mask-heatmaps",
            dest="show_mask_heatmaps",
            help="Show a heatmap probability for the top masks-per-dim masks",
            action="store_true",
        )
        parser.add_argument(
            "--masks-per-dim",
            type=int,
            default=2,
            help="Number of heatmaps per dimension to show",
        )
        parser.add_argument(
            "opts",
            help="Modify model config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )

        args = parser.parse_args()

        # load config from file and command-line arguments
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        self.cfg = cfg
        # prepare object that handles inference plus adds predictions on top of image
        self.coco_demo = COCODemo(
            cfg,
            confidence_threshold=args.confidence_threshold,
            show_mask_heatmaps=args.show_mask_heatmaps,
            masks_per_dim=args.masks_per_dim,
            min_image_size=args.min_image_size,
        )

    def get_cfg(self):
        return self.cfg

    def run(self, img, resize = True):
        predictions, ponder_cost = self.coco_demo.compute_prediction(img, resize)
        bbox = self.coco_demo.select_top_predictions(predictions)
        bbox_complexity = pondercost_proc.ponder_cost_postproc(bbox.bbox, ponder_cost, img.shape)

        # pondercost_proc.vis_ponder_cost(ponder_cost)
        return bbox, bbox_complexity


    def mask_overlay(self, img, mask, dist=False):
        return self.coco_demo.overlay(img, mask, dist)

def main():
    cam = cv2.VideoCapture(0)
    mask_engine = MaskCompute()
    filter_actor = Filter()

    partition_mgr = PartitionManager(parition_num=2)
    prediction_mgr = PredictionManager(mask_engine.get_cfg())
    control_mgr = ControlManager(prediction_mgr, partition_mgr, mask_engine, '/home/wuyang/maskrcnn-benchmark/state_monitor/server.conf')
    e2e_latency = []

    while True:
        # counting the time when receiving the frame.
        start_time = time.time()
        ret_val, img = cam.read()

        #todo[Priority:low]: need to handle when the first frame has been skipped
        img = filter_actor.filter(img)
        composite = None
        if img is not None:
            branch = control_mgr.get_curr_state()
            # cold start: run locally
            if branch == 'cold_start':
                bbox, units = mask_engine.run(img)
                prediction_mgr.add_mask(bbox, units)

            elif branch == 'distribute':
                bbox, weight = prediction_mgr.next_predict_workload_dist()

                # get computing capability of involved nodes
                resources = control_mgr.report_resources()

                # perform frame partition based on computing capability of involved nodes and computing overheads of jos
                imgs = partition_mgr.frame_partition(img, bbox, weight, resources)

                # offload the partitions to edge servers
                control_mgr.distribute(imgs)

                # merge all the results
                composite, bbox, units = control_mgr.merge_partitions()

                prediction_mgr.add_mask(bbox, units)

            elif control_mgr.branch_states[branch] == 'shortcut':
                composite = None

            control_mgr.last_composite = composite
            cv2.imshow("COCO detections", composite)
        else:
            cv2.imshow("COCO detections", control_mgr.last_composite)

        latency = time.time()-start_time
        print("Time: {:.2f} s / img".format(latency))
        e2e_latency.append(latency)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
