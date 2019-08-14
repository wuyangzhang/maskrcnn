import time, pickle

import sys

sys.path.append('/home/nvidia/maskrcnn-benchmark')

import cv2
from app import ApplicationManager
from config import Config
from state_monitor import RemoteConnector
import logging

log_output_path = './single_offload.log'

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=log_output_path,
                    level=logging.INFO,
                    )

log = logging.getLogger('maskrcnn')

def main():
    config = Config()
    paths = []
    with open('eval/mask_rcnn/davis_videos-nvidia.txt', 'r') as f:
        for line in f.readlines():
            paths.append(line.split()[0])

    ip = '192.168.55.100'
    port = 5051
    remote_con = RemoteConnector(id, ip, port)
    remote_con.log = log

    for img_path in paths:

        log.info(img_path)

        path = img_path.split('/')[-2]
        log.info('path,{}'.format(path))
        print(path)
        img = cv2.imread(img_path)

        # offloading & get results back!

        if config.frame_width is None:
            config.frame_height, config.frame_width = img.shape[:2]

        log.info('offload size,{}'.format(img.shape))
        # counting the time when receiving the frame

        remote_con.send(img, [])

        time.sleep(0.03)


cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
