import time, pickle
import sys
sys.path.append('/home/nvidia/maskrcnn-benchmark')

import cv2
from config import Config
from app import ApplicationManager
from dataset import MobiDistDataset


if __name__ == "__main__":

    app_mgr = ApplicationManager()
    config = Config()
    dataset = MobiDistDataset(config).getDataLoader()
    resolution = [(224, 224), (640, 480), (1280, 720), (2048, 1080), (3840, 2160)]


    for id, img in enumerate(dataset):

        img = img[0].numpy()
        #((375, 1242, 3))
        for w, h in resolution:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

            for _ in range(5):
                start = time.time()
                bbox, composite = app_mgr.run(img)
                print('time*{}*resolutiuon with *{}'.format(time.time()-start, (w, h)))
