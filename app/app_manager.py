import argparse
import threading
from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
import cv2
import matplotlib.pyplot as plt

lock = threading.Lock()


class ApplicationManager:
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

    def run(self, img, res=None, resize=True, display=False):
        '''
        Given each application with SACT, we require it to output 3 results,
        1) final application rendering results
        2) bbox coordinates
        3) bbox computing complexity
        '''

        predictions, overhead = self.coco_demo.compute_prediction(img, resize)
        #print(len(predictions.bbox))
        bbox = self.coco_demo.select_top_predictions(predictions)
        composite = self.coco_demo.overlay_boxes(img, bbox)
        #composite = self.coco_demo.overlay_mask(img, bbox)

        if display:
            lock.acquire()
            b, g, r = cv2.split(composite)
            composite = cv2.merge([r, g, b])
            plt.imshow(composite)
            plt.show()
            lock.release()

        # cv2.imshow("distributed s input", img)
        # cv2.waitKey(0)
        #cv2.imshow("distributed s res ", composite)
        #cv2.waitKey(2)

        if res is not None:
            res.append(bbox)
        return bbox, composite

    def mask_overlay(self, img, mask, dist=False):
        return self.coco_demo.overlay(img, mask, dist)

    def rendering(self, img, bbox):
        #return self.coco_demo.overlay_mask(img, bbox)
        return self.coco_demo.overlay_boxes(img, bbox)
        #return self.coco_demo.overlay_boxes_only(img, bbox)
