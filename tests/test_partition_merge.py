
import cv2
from demo.predictor import COCODemo
import argparse
from maskrcnn_benchmark.config import cfg
from partition.partition_manager import PartitionManager
from demo.mobile_client import ApplicationManager
import matplotlib.pyplot as plt
from config import Config




# engine = ApplicationManager()
#
# bbox, bbox_complexity = engine.run_on_opencv_image(img)
# composite = engine.mask_overlay(img, bbox)

parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
parser.add_argument(
    "--config-file",
    default="./configs/e2e_mask_rcnn_R_50_FPN_1x.yaml",
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

# prepare object that handles inference plus adds predictions on top of image
coco_demo = COCODemo(
    cfg,
    confidence_threshold=args.confidence_threshold,
    show_mask_heatmaps=args.show_mask_heatmaps,
    masks_per_dim=args.masks_per_dim,
    min_image_size=args.min_image_size,
)

#file = 'COCO_test2014_000000032246.jpg' #baseball player
file = 'COCO_test2014_000000005771.jpg' #clock


def test(img_addr):
    img = cv2.imread(img_addr)
    src = img.copy()
    src1 = img.copy()

    composite, bbox, units = coco_demo.run_on_opencv_image(img)
    composite = coco_demo.overlay_boxes(img, bbox)
    b, g, r = cv2.split(composite)  # get b,g,r
    label_image = cv2.merge([r, g, b])  #

    config = Config()
    config.frame_width = img.shape[1]
    config.frame_height = img.shape[0]

    partition_num = config.par_num

    partition_mgr = PartitionManager(config)

    imgs = partition_mgr.frame_partition(src1, bbox.bbox, [1] * len(bbox.bbox), [1] * partition_num)

    b, g, r = cv2.split(imgs[0])  # get b,g,r
    i0 = cv2.merge([r, g, b])
    plt.imshow(i0)
    plt.show()

    b, g, r = cv2.split(imgs[1])  # get b,g,r
    i1 = cv2.merge([r, g, b])
    plt.imshow(i1)
    plt.show()

    res = []
    for i, img in enumerate(imgs):

        composite, bbox, units = coco_demo.run_on_opencv_image(img)
        res.append(bbox)

        b, g, r = cv2.split(composite)  # get b,g,r
        tmp = cv2.merge([r, g, b])

        plt.imshow(tmp)
        plt.show()

    bbox = partition_mgr.merge_partition(res)

    res = coco_demo.overlay_boxes(src, bbox)
    b, g, r = cv2.split(res)  # get b,g,r
    i1 = cv2.merge([r, g, b])
    plt.imshow(i1)
    plt.show()


    plt.imshow(label_image)
    plt.show()


if __name__ == '__main__':

    # test a single image
    '''
    /home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/car-turn/00037.jpg

    /home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/car-turn/00041.jpg
    '''
    test('/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p/car-turn/00037.jpg')

    import random
    random.seed(0)
    with open('eval/mask_rcnn/davis_videos.txt', 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            print(line)
            test(line.split()[0])