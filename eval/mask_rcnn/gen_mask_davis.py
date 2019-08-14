import time, pickle, glob, os
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


    print('finish loading')
    # init the main components of MobiDist
    config = Config()

    time.sleep(3)
    dataset = MobiDistDataset(config).getDataLoader()

    res = dict()
    print('running dataset')
    # get gt list

    gt_list = list()
    for gt in glob.glob('eval/mask_rcnn/davis_gt_*.pkl'):
        gt_name = gt.split('.')[0].split('_')[-1]
        gt_list.append(gt_name)


    app_mgr = ApplicationManager()

    # get all
    root_path = '/home/wuyang/datasets/davis/DAVIS/JPEGImages/480p'
    for gt in gt_list:
        path = root_path + '/' + gt
        imgs = os.listdir(path)
        imgs.sort()
        masks = list()

        while True:
            for i in imgs:
                img = cv2.imread(path + '/' + i)
                bbox, composite = app_mgr.run(img)
                masks.append(bbox.extra_fields['mask'])
                time.sleep(0.2)

        with open('eval/mask_rcnn/davis{}_non_par.pkl'.format(gt), 'wb') as f:
            pickle.dump(masks, f)

    '''
    for id, img in enumerate(dataset):
        img, subject = img
        if subject in gt_list:
            pass
        start = time.time()
        img = img[0].numpy()
        bbox, composite = app_mgr.run(img)

        mask = bbox.extra_fields['mask']
        res[id] = mask

        if id > 2 and id % 5 == 0:
            with open('eval/mask_rcnn/davis{}_non_par.pkl'.format(id), 'wb') as f:
                pickle.dump(res, f)
                res = dict()
    '''
        #a = 1
        # b, g, r = cv2.split(composite)  # get b,g,r
        # pred_image = cv2.merge([r, g, b])  # switch it to rgb
        # plt.imshow(pred_image)
        # plt.show()
        #
        # print(time.time()-start)
