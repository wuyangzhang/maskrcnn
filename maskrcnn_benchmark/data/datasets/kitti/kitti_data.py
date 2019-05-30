import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image


class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):
        indices = []
        for i in range(len(end_idx) - 1):
            start = end_idx[i]
            end = end_idx[i + 1] - seq_length
            indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


class KITTIDataset(Dataset):
    def __init__(self, image_paths, seq_length, transform, length):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length

    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        #print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        path = [self.image_paths[i][0] for i in indices]
        first = path[0].split('/')[-2]
        last = path[-1].split('/')[-2]

        if first != last:
            print('Will deliever the following path... {}'.format(path))
        images = []
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        x = torch.stack(images)

        # prediction target
        target = Image.open(self.image_paths[i+1][0])
        if self.transform:
            target = self.transform(target)
        y = torch.tensor(target)

        return x, y

    def __len__(self):
        return self.length


def get_loader(seq_length=5, batch_size=1):

    root_dir = '/home/wuyang/data_tracking/training/'
    class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]
    class_image_paths = []
    end_idx = []
    for c, class_path in enumerate(class_paths):
        for d in os.scandir(class_path):
            if d.is_dir:
                paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
                # Add class idx to paths
                paths = [(p, c) for p in paths]
                class_image_paths.extend(paths)
                end_idx.extend([len(paths)])

    end_idx = [0, *end_idx]
    end_idx = torch.cumsum(torch.tensor(end_idx), 0)


    sampler = MySampler(end_idx, seq_length)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = KITTIDataset(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(sampler)
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True
    )

    return loader

# for data, target in loader:
#     #print(data.shape)
#     a = 1