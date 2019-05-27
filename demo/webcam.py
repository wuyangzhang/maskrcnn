# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
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

    # cam = cv2.VideoCapture(0)
    # while True:
    #     start_time = time.time()
    #     ret_val, img = cam.read()
    #     composite = coco_demo.run_on_opencv_image(img)
    #     print("Time: {:.2f} s / img".format(time.time() - start_time))
    #     cv2.imshow("COCO detections", composite)
    #     if cv2.waitKey(1) == 27:
    #         break  # esc to quit
    # cv2.destroyAllWindows()

    cap = cv2.VideoCapture('/home/wuyang/maskrcnn-benchmark/demo/project.avi')

    fps = 30

    seconds = 60 * 5

    total_frame = fps * seconds

    img1 = cap.read()[1]

    height, width, layers = img1.shape

    size = (width, height)
    #
    # while (cap.isOpened()):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     if ret == True:
    #
    #         # Display the resulting frame
    #         cv2.imshow('Frame', frame)
    #
    #         # Press Q on keyboard to  exit
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break
    #
    #     # Break the loop
    #     else:
    #         break
    #
    # # When everything done, release the video capture object
    # cap.release()


    video = cv2.VideoWriter('project-mask.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)


    i = 1

    latency = []
    import time
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            #cv2.imshow('Frame', frame)
            start = time.time()
            composite = coco_demo.run_on_opencv_image(frame)
            cost = time.time() - start
            latency.append(cost)
            print(cost)
            cv2.imshow("COCO detections", composite)
            video.write(composite)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q') or i == 3600:
                break
            print(i)
            i+=1
        # Break the loop
        else:
            break

    with open('your_file.txt', 'w') as f:
        for item in latency:
            f.write("%s\n" % item)

    # When everything done, release the video capture object
    cap.release()
    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
