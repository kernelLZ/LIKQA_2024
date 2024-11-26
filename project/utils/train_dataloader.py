# -*- coding: utf-8 -*-
# @Time    : 2024/9/19 17:16
# @Author  : BitYang
# @FileName: train_dataloader.py
# @Software: PyCharm

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
# from conda.exports import root_dir
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MultiColorSpaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.annotations = pd.concat([df, df], ignore_index=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]))
        image = Image.open(img_name).convert('RGB')
        rgb_image = image.convert('RGB')
        hsv_image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV))
        lab_image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB))
        yuv_image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV))

        if self.transform:
            rgb_image = self.transform(rgb_image)
            hsv_image = self.transform(hsv_image)
            lab_image = self.transform(lab_image)
            yuv_image = self.transform(yuv_image)

        annotations = (self.annotations.iloc[idx, 1:]).to_numpy()
        annotations = annotations.astype('float').reshape(-1, 1)

        sample = {
            'img_id': img_name,
            'RGB_Image': rgb_image,
            'HSV_Image': hsv_image,
            'LAB_Image': lab_image,
            'YUV_Image': yuv_image,
            'annotations': annotations
        }

        return sample


class MYDatasetLoader:
    def __init__(self, datasets, batch_size, shuffle=True, num_workers=8):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers) for dataset in datasets]
        self.iterators = [iter(loader) for loader in self.loaders]

    def __iter__(self):
        self.iterators = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        if not self.iterators:
            raise StopIteration
        current_loader = np.random.choice(self.iterators)
        try:
            return next(current_loader)
        except StopIteration:
            self.iterators.remove(current_loader)
            if not self.iterators:
                raise StopIteration
            return next(self)

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)


def build_dataset(batch_size, csv_files, root_dirs):
    """
    The method is used build dataset for utils model.
    :param batch_size: batch_size,
    :param csv_files: data file,    :param root_dirs: file root.
    :return:
    """
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datasets = [MultiColorSpaceDataset(csv_file=csv_files, root_dir=root_dirs, transform=transform)]
    loader = MYDatasetLoader(datasets, batch_size)
    return loader


if __name__ == "__main__":
    batch_size = 4
    file = 'training3.csv'
    dir = '../datasets/'
    loader = build_dataset(batch_size, file, dir)
    for _, data in enumerate(loader):
        print(data)
    print(loader)