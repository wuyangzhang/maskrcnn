import os



import glob

import cv2

#from app.app_manager import ApplicationManager

from config import Config

config = Config()

root_dir = config.home_addr + 'kitty/testing/image_02/'
output_dir = config.home_addr + 'kitty/testing/seq_list.txt'

# from demo.mobile_client import MaskCompute


"""
Calculate bbox and complexity of all images.
Write results to files.
Need to normalize bbox.. and complexity
"""


def cal_bbox(folderpath):
    mask_compute = ApplicationManager()

    for folder in os.listdir(folderpath):
        for img_path in glob.glob(folderpath + '/' + folder + '/*.png'):

            bbox_file_path = img_path.split('.')[0] + '.txt'
            img = cv2.imread(img_path)
            _, bbox, bbox_complexity = mask_compute.run(img)
            print(bbox)
            height, width = img.shape[:2]
            print('processing img {}'.format(img_path))
            with open(bbox_file_path, 'w+') as f:
                for i in range(len(bbox)):
                    vals = bbox.bbox[i].data.tolist()
                    vals.append(bbox_complexity[i].data.tolist())
                    vals[0] /= width
                    vals[1] /= height
                    vals[2] /= width
                    vals[3] /= height
                    vals[4] /= 50
                    vals = ' '.join(str(val) for val in vals)
                    f.write(vals + '\n')
    print('finish bbox calculation')


def process():
    f = open(output_dir, 'w')
    for video in os.listdir(root_dir):
        video_dir = root_dir + video
        index = []
        for v in glob.glob(video_dir + '/*.txt'):
            v = v.split('/')[-1]
            index.append(int(v.split('.')[0]))
        max_index, min_index = max(index), min(index)
        f.write('{},{}\n'.format(video_dir, max_index + 1))

    f.close()

# cal_bbox(root_dir)
process()
