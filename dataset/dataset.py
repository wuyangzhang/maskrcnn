import random
from torch.utils.data import Dataset, DataLoader
import cv2

class MobiDistDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset = config.eval_dataset

        if self.dataset == 'kitti':
            self.files = config.kitti_video_path  # seq_list.txt
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
            if start_video_index >= self.video_size[l] - 1 or start_video_index < 1:
                start_video_index = random.randint(1, self.video_size[l] - 2)

            input_path = select_data + '/' + '0' * (6 - len(str(start_video_index))) + str(start_video_index) + '.png'

            return self.load_tensor(input_path)

        else:
            raise NotImplementedError

    def getDataLoader(self):
        return DataLoader(self, batch_size=1, shuffle=False)

    @staticmethod
    def load_tensor(filepath):
        return cv2.imread(filepath)