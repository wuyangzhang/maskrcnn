import random, os.path, glob
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

        elif self.dataset == 'davis':

            self.files = config.davis_video_path + 'seq_list.txt'
            if not os.path.exists(self.files):
                with open(self.files, 'w') as f:
                    for video in os.listdir(config.davis_video_path):
                        if video == 'seq_list.txt':
                            continue
                        video = config.davis_video_path + video
                        cnt = len(os.listdir(video))
                        f.write(video + ',' + str(cnt) + '\n')

            with open(self.files, 'r') as f:
                self.video_size = []
                self.video_dir = []
                for line in f.readlines():
                    video_dir, size = line.split(',')
                    self.video_dir.append(video_dir)
                    self.video_size.append(int(size))

                # calculate prefix sum
                self.prefix = [self.video_size[0]] * len(self.video_size)

                for i in range(1, len(self.video_size)):
                    self.prefix[i] = self.video_size[i] + self.prefix[i - 1]

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

            select_video_dir = self.video_dir[l]
            if l < 1:
                start_video_index = index
            else:
                start_video_index = index - self.prefix[l - 1]

            # corner case: cannot find sufficient preceding or next video frames
            if start_video_index >= self.video_size[l] - 1 or start_video_index < 1:
                start_video_index = random.randint(1, self.video_size[l] - 2)

            input_path = select_video_dir + '/' + '0' * (6 - len(str(start_video_index))) + str(start_video_index) + '.png'
            return self.load_tensor(input_path)

        elif self.dataset == 'davis':
            l, r = 0, len(self.video_size)
            while l < r:
                m = l + (r - l) // 2
                if self.prefix[m] < index:
                    l = m + 1
                else:
                    r = m

            if index == self.prefix[l]:
                l += 1
            select_video_dir = self.video_dir[l]
            if l < 1:
                start_video_index = index
            else:
                start_video_index = index - self.prefix[l - 1]

            input_path = select_video_dir + '/' + '0' * (5 - len(str(start_video_index))) + str(start_video_index) + '.jpg'
            print(input_path)
            return self.load_tensor(input_path), input_path.split('/')[-2]

        else:
            raise NotImplementedError

    def getDataLoader(self):
        return DataLoader(self, batch_size=1, shuffle=False)

    @staticmethod
    def load_tensor(filepath):
        return cv2.imread(filepath)



if __name__ == '__main__':
    from config import Config
    config = Config()
    dataset = MobiDistDataset(config)
    for img, id in dataset:
        b = 1
        cv2.imshow('dataloader', img)
        cv2.waitKey(1)
