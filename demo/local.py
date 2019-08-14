import time, pickle
import sys
sys.path.append('/home/nvidia/maskrcnn-benchmark')

import cv2
import matplotlib.pyplot as plt

from config import Config
from app import ApplicationManager
from dataset import MobiDistDataset


if __name__ == "__main__":

    load = False
    if load:
        f = open('demo/gt/kitti.pkl', 'rb')
        dict = pickle.load(f)
        f.close()
        exit(0)

    app_mgr = ApplicationManager()

    print('finish loading')
    # init the main components of MobiDist
    config = Config()

    time.sleep(3)
    dataset = MobiDistDataset(config).getDataLoader()

    #res = dict()
    print('running dataset')
    for id, img in enumerate(dataset):
        #print('working on id', id)
        start = time.time()
        img = img[0].numpy()
        bbox, composite = app_mgr.run(img)


        #a = 1
        # b, g, r = cv2.split(composite)  # get b,g,r
        # pred_image = cv2.merge([r, g, b])  # switch it to rgb
        # plt.imshow(pred_image)
        # plt.show()
        #
        # print(time.time()-start)
