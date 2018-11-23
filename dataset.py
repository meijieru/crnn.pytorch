#!/usr/bin/python
# encoding: utf-8
import pathlib
import random
from PIL import Image
import numpy as np
import six
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler
import cv2


def cv_imread(file_path: str, flag: int = 1) -> np.array:
    """
    使用 opencv 读取中文路径的图片
    :param file_path: 文件路径
    :param flag: 读取方式 1：彩色图，0L灰度图
    :return: 图片 array
    """
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)
    return cv_img


class ImageDataset(Dataset):
    def __init__(self, data_txt: str, data_shape: tuple, img_channel: int, phase: str = 'train',
                 img_type='cv', transform=None, target_transform=None):
        """
        数据集初始化
        :param data_txt: 存储着图片路径和对于label的文件
        :param data_shape: 图片的大小(h,w)
        :param img_channel: 图片通道数
        :param alphabet: 字母表
        """
        super(ImageDataset, self).__init__()
        assert phase in ['train', 'test'] and img_type in ['cv', 'PIL']

        self.data_list = []
        with open(data_txt, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                img_path = pathlib.Path(line[0])
                if img_path.exists() and img_path.stat().st_size > 0 and line[1]:
                    self.data_list.append((line[0], line[1]))
        self.img_h = data_shape[0]
        self.img_w = data_shape[1]
        self.img_channel = img_channel
        self.phase = phase
        self.img_type = img_type
        self.transform = transform
        self.target_transform = target_transform
        self.label_dict = {}

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        label = label.replace(' ', '')
        if self.img_type == 'cv':
            img = self.pre_processing(img_path)
        else:
            img = self.pre_processing_pil(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.data_list)

    def pre_processing(self, img_path):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度

        :param img_path: 图片地址
        :return:
        """
        data_augment = False
        if self.phase == 'train' and np.random.rand() > 0.5:
            data_augment = True
        if data_augment:
            img_h = 40
            img_w = 340
        else:
            img_h = self.img_h
            img_w = self.img_w
        img = cv_imread(img_path, 1 if self.img_channel == 3 else 0)
        h, w = img.shape[:2]
        ratio_h = float(img_h) / h
        new_w = int(w * ratio_h)
        if new_w < img_w:
            img = cv2.resize(img, (new_w, img_h))
            step = np.zeros((img_h, img_w - new_w, self.img_channel), dtype=img.dtype)
            img = np.column_stack((img, step))
        else:
            img = cv2.resize(img, (img_w, img_h))
        if data_augment:
            # random crop
            starty = np.random.randint(0, img_h - self.img_h + 1)
            startx = np.random.randint(0, img_w - self.img_w + 1)
            img = img[starty:, startx:]
        return img

    def pre_processing_pil(self, img_path):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度

        :param img_path: 图片地址
        :return:
        """
        data_augment = False
        if self.phase == 'train' and np.random.rand() > 0.5:
            data_augment = True
        if data_augment:
            img_h = 40
            img_w = 340
        else:
            img_h = self.img_h
            img_w = self.img_w
        img = Image.open(img_path)
        if self.img_channel == 1:
            img = img.convert('L')

        w, h = img.size
        ratio_h = float(img_h) / h
        new_w = int(w * ratio_h)
        if new_w < img_w:
            img = img.resize((new_w, img_h))
            img_pad = Image.new("RGB" if self.img_channel == 3 else "L", (img_w, img_h))
            img_pad.paste(img)
            img = img_pad
        else:
            img = img.resize((img_w, img_h))
        if data_augment:
            # random crop
            img = transforms.RandomCrop((self.img_h, self.img_w))(img)
        return img


if __name__ == '__main__':
    import keys
    import time
    import config
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    from torchvision.transforms import ToTensor
    from predict import decode
    import utils

    converter = utils.strLabelConverter(config.alphabet)

    font = FontProperties(fname=r"simsun.ttc", size=14)
    alphabet = keys.txt_alphabet
    dataset = ImageDataset(config.trainfile, (32, 320), 3, 'train', 'PIL', ToTensor())
    data_loader = DataLoader(dataset, 1, shuffle=True, num_workers=1)
    print(len(dataset))
    start = time.time()
    for i, (img, label, img_path) in enumerate(data_loader):
        targets, targets_lengths = converter.encode(label)
        targets = torch.Tensor(targets).int()
        targets_lengths = torch.Tensor(targets_lengths).int()
        if torch.isnan(img).any().item() or torch.isnan(targets).any().item() or torch.isnan(
                targets_lengths).any().item():
            print(i, torch.isnan(img).any().item(), torch.isnan(targets).any().item(),
                  torch.isnan(targets_lengths).any().item())
