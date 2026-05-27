import numpy as np
import torch
from torch.utils.data import Dataset
import os
import imageio
from . import imutils

def load_img_name_list(img_name_list_path):
    with open(img_name_list_path, 'r') as f:
        img_name_list = f.read().splitlines()
    return img_name_list

class FMBDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        split='train',
        stage='train',
        resize_range=[512, 640],
        rescale_range=[0.5, 2.0],
        crop_size=512,
        img_fliplr=True,
        ignore_index=255,
        aug=False,
        **kwargs
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'Infrared')
        self.label_dir = os.path.join(root_dir, 'Label')
        self.name_list_path = os.path.join(root_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_path)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = imutils.PhotoMetricDistortion()

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            if self.rescale_range:
                image, label = imutils.random_scaling(
                    image,
                    label,
                    scale_range=self.rescale_range,
                    size_range=self.resize_range)
            if self.img_fliplr:
                image, label = imutils.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = imutils.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    mean_rgb=[123.675, 116.28, 103.53],
                    ignore_index=self.ignore_index)
        
        if self.stage != "train":
            image = imutils.img_resize_short(image, min_size=min(self.resize_range))

        image = imutils.normalize_img(image)
        # to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        name = self.name_list[idx]
        
        # 读取图像
        img_path = os.path.join(self.img_dir, name)
        image = np.asarray(imageio.imread(img_path))
        if len(image.shape) == 2:  # 如果是单通道图像
            image = np.stack([image] * 3, axis=-1)  # 转换为3通道
            
        # 读取标签
        label_path = os.path.join(self.label_dir, name)
        if self.stage in ["train", "val"]:
            label = np.asarray(imageio.imread(label_path))
        else:
            label = np.zeros_like(image[:,:,0])  # 测试阶段使用空标签

        image, label = self.__transforms(image=image, label=label)

        return name, image, label 