'''
We prepare the dataset by converting
video frames into the bbox formats: [x1, y1, x2, y2, complexity].
Each frame contains a set of the bbox coordinates and computing
complexity.

We run MaskRCNN over video datasets to extract those information.
We consider the following video datasets:
UCF-101 Datasets
Human3.6M Datasets
CityScape Datasets
KTH Datasets
Robotic Pushing Datasets
'''
import glob
import os
import random

import numpy as np
import cv2
import torch
import torch.nn
import torch.nn.utils.rnn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class RPPNDataset(Dataset):

    def __init__(self, video_files, dataset, window=5, simulate=False, batch_size=64):
        """Constructor
        :param video_files: the root folder of video files
        :param window: the size of sliding window
        :param simulate: whether to use simulated data
        :param batch_size: batch size
        :param max_padding_len: the max number of region proposal in an image
        """

        self.window = window
        self.batch_size = batch_size
        self.dataset = dataset
        self.cnt = 0
        self.max_padding_len = 30

        if simulate:
            self.data = get_example_output(batch_size)
            return

        if dataset == 'kitti':
            self.files = video_files  # seq_list.txt
            f = open(self.files, 'r')
            self.video_size = []
            self.video_dir = []
            for line in f.readlines():
                video_dir, size = line.split(',')
                self.video_size.append(int(size))
                self.video_dir.append(video_dir)

            # calculate prefix sum
            self.prefix = [self.video_size[0]] * len(self.video_size)

            for i in range(1, len(self.video_size)):
                self.prefix[i] = self.video_size[i] + self.prefix[i - 1]

            f.close()
        else:
            raise NotImplementedError

    def __len__(self):
        return self.prefix[-1]

    def __getitem__(self, index):
        if self.dataset == 'kitti':

            l, r = 0, len(self.video_size)
            while l < r:
                m = l + (r - l) // 2
                if self.prefix[m] < index:
                    l = m + 1
                else:
                    r = m

            select_data = self.video_dir[l]
            if l < 1:
                start_video_index = index
            else:
                start_video_index = index - self.prefix[l - 1]

            # corner case: cannot find sufficient preceding or next video frames
            if start_video_index >= self.video_size[l] - 1 or start_video_index < self.window:
                start_video_index = random.randint(self.window, self.video_size[l] - 2)

            # find preceding video frames with slide window
            indexes = [start_video_index - i for i in range(self.window)]
            input_path = [select_data + '/' + '0' * (6 - len(str(i))) + str(i) + '.txt' for i in indexes]
            input_tensors = []

            # format 1: input shape = batch_size, seq_length (total bbox number), 5 features
            # for path in input_path:
            #     input_tensors += self.load_tensor(path, self.max_padding_len)
            # input_tensors = torch.as_tensor(input_tensors).reshape(-1, 5)

            # format 2: input shape = batch_size, seq_length (total frame number), 5 features * 30 bbox/frame. Designed for LSTM input
            for path in input_path:
                input_tensors.append(torch.tensor(self.load_tensor(path, self.max_padding_len)).reshape(-1))
            input_tensors = torch.stack(input_tensors)

            target_path = select_data + '/' + '0' * (6 - len(str(start_video_index + 1))) + str(
                start_video_index + 1) + '.txt'
            target_tensor = self.load_tensor(target_path, self.max_padding_len, padding=True)
            target_tensor = torch.as_tensor(target_tensor).reshape(-1, 5)
            # target_tensor = torch.flatten(target_tensor)
            # input_tensors = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=True)
            return input_tensors, target_tensor

        else:
            raise NotImplementedError

    def getDataLoader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def load_tensor(filepath, max_padding_len, padding=True):
        """
        load bbox's coordinates and computing complexity in a sing image.
        When the number of bbox is smaller than the max length, we pad
        at the end.
        :param filepath: image path
        :return a list in the shape of max_length x 5
        """
        with open(filepath, 'r') as f:
            res = []
            for line in f.readlines():
                vals = line.split(' ')
                vals = [float(val) for val in vals]
                res.append(vals)
            if padding:
                for _ in range(max_padding_len - len(res)):
                    res.append([0.0] * 5)
            return res


def get_example_output(batch=100):
    res = []
    for _ in range(batch):
        # randomly generate the number of region proposals in a single frame
        rp_num = np.random.randint(0, 10)
        # randomly generate the normalized coordinates and the computing complexity
        single = np.random.rand(rp_num, 5)
        single = torch.tensor(single)
        res.append(single)
    # important: padding zero region proposals to the frame.
    return torch.nn.utils.rnn.pad_sequence(res, batch_first=True)


def make_dataset(X, Y):
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def cal_IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def slide_window(x, look_back, stride=1):
    X, Y = [], []
    for i in range(0, len(x) + 1 - look_back, stride):
        X.append(np.concatenate(x[i: i + look_back]))
        Y.append(x[i + 1])
    return X, Y


# dataset = get_example_output()
# X, Y = slide_window(dataset, look_back)
# X_train, Y_train, X_test, Y_test = make_dataset(X, Y)

# cal_bbox('/home/wuyang/kitty/training/image_02')

def test():
    video_files = '/home/wuyang/kitty/testing/seq_list.txt'
    dataset = 'kitti'
    data_loader = RPPNDataset(video_files, dataset).getDataLoader(shuffle=False)

    for data in data_loader:
        print(data)
