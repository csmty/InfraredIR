import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from albumentations import (
    GaussNoise, Lambda, GaussianBlur, Compose, RandomCrop, RandomBrightnessContrast
)
from torch.utils.data.dataset import Dataset
import os

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


def stripe_noise(image, **params):
    if random.random() < 0.5:
        g = np.random.randn(1, image.shape[1]) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 5)
    else:
        g = np.random.randn(image.shape[0], 1) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 5)
    if len(image.shape) == 3:
        g = np.expand_dims(g, -1)
        b = np.expand_dims(b, -1)
    noise = image * g + b
    image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
    return image


def nonuniformity_optical(image, **params):
    # print(image.shape,"==============")
    h, w, _ = image.shape
    noise = np.ones((h, w)).astype("float32")
    idx_h = np.expand_dims(np.arange(1, h + 1), 1)
    idx_w = np.expand_dims(np.arange(1, w + 1), 0)
    delta = np.random.randint(15, 75 + 1)
    ch = np.random.randint(h)
    cw = np.random.randint(w)

    p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
    p /= np.max(p)
    noise *= p
    noise = np.cos(noise * np.pi / 2) ** 4
    if len(image.shape) == 3:
        noise = np.expand_dims(noise, -1)
    if random.random() < 0.5:
        image = np.clip(image.astype("float32") + noise.astype("float32") * delta, 0, 255).astype("uint8")
    else:
        image = np.clip(image.astype("float32") + (1 - noise.astype("float32")) * delta, 0, 255).astype("uint8")
    return image


def LC(p=1):
    return RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(-0.8, -0.2), p=p)


def Blur(p=1):
    return Compose([
        GaussianBlur(blur_limit=(7, 23), sigma_limit=(1, 3), p=1),
    ], p=p)


def Noise(p=1):
    return Compose([
        Lambda(image=nonuniformity_optical, p=1),
        Lambda(image=stripe_noise, p=1),
        GaussNoise(var_limit=(5.0 ** 2, 20.0 ** 2), p=1),
    ], p=p)


def deg_simple(p=1):
    return Compose([
        # step 1
        RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(-0.8, -0.2), p=0.8),
        # step 2
        GaussianBlur(blur_limit=(7, 23), sigma_limit=(1, 3), p=0.8),
        # step 3
        Lambda(image=nonuniformity_optical, p=0.8),
        Lambda(image=stripe_noise, p=0.8),
        # GaussNoise(var_limit=(5.0 ** 2, 20.0 ** 2), p=0.8),
        GaussNoise(std_range=(5.0 /255.0, 20.0 /255.0), p=0.8),
        # GaussNoise(std_range=(0.02, 0.08), p=1.0)
    ], p=p)


def deg_single(p=1):
    r = random.random()
    if r < 0.33:
        return Noise(p=p)
    elif r < 0.66:
        return Blur(p=p)
    else:
        return LC(p=p)

def degradation(p=1):
    if random.random() < 0.5:
        return deg_simple(p=p)
    else:
        return deg_single(p=p)


# def stripe_noise(image, **params):
#     # 控制退化的程度
#     intensity = params.get('intensity', 1.0)  # 默认退化强度为 1.0
#     if random.random() < 0.5:
#         g = np.random.randn(1, image.shape[1]) * (np.random.rand() * 0.07 + 0.03) * intensity
#         b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 5) * intensity
#     else:
#         g = np.random.randn(image.shape[0], 1) * (np.random.rand() * 0.07 + 0.03) * intensity
#         b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 5) * intensity
#     if len(image.shape) == 3:
#         g = np.expand_dims(g, -1)
#         b = np.expand_dims(b, -1)
#     noise = image * g + b
#     image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
#     return image


# def nonuniformity_optical(image, **params):
#     # 控制退化的程度
#     intensity = params.get('intensity', 2.0)  # 提高退化强度
#     h, w, _ = image.shape
#     noise = np.ones((h, w)).astype("float32")
#     idx_h = np.expand_dims(np.arange(1, h + 1), 1)
#     idx_w = np.expand_dims(np.arange(1, w + 1), 0)
#     delta = np.random.randint(15, 75 + 1)
#     ch = np.random.randint(h)
#     cw = np.random.randint(w)

#     p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
#     p /= np.max(p)
#     noise *= p
#     noise = np.cos(noise * np.pi / 2) ** 4
#     if len(image.shape) == 3:
#         noise = np.expand_dims(noise, -1)
#     if random.random() < 0.5:
#         image = np.clip(image.astype("float32") + noise.astype("float32") * delta * intensity, 0, 255).astype("uint8")
#     else:
#         image = np.clip(image.astype("float32") + (1 - noise.astype("float32")) * delta * intensity, 0, 255).astype("uint8")
#     return image


# def LC(p=1, **params):
#     brightness_limit = params.get('brightness_limit', (0.4, 0.6))  # 增强亮度变化
#     contrast_limit = params.get('contrast_limit', (-1.0, -0.4))  # 增强对比度变化
#     return RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=p)


# def Blur(p=1, **params):
#     blur_limit = params.get('blur_limit', (15, 40))  # 增加模糊范围
#     sigma_limit = params.get('sigma_limit', (2, 5))  # 增强模糊强度
#     return Compose([
#         GaussianBlur(blur_limit=blur_limit, sigma_limit=sigma_limit, p=1),
#     ], p=p)


# def Noise(p=1, **params):
#     return Compose([
#         Lambda(image=nonuniformity_optical, p=1, **params),
#         Lambda(image=stripe_noise, p=1, **params),
#         GaussNoise(var_limit=(10.0 ** 2, 50.0 ** 2), p=1),  # 提高噪声强度
#     ], p=p)


# def deg_simple(p=1, **params):
#     intensity = params.get('intensity', 2.0)  # 增强退化强度
#     return Compose([
#         # step 1
#         RandomBrightnessContrast(brightness_limit=(0.2, 0.6), contrast_limit=(-1.0, -0.4), p=0.8),  # 增强亮度和对比度变化
#         # step 2
#         GaussianBlur(blur_limit=(15, 40), sigma_limit=(2, 5), p=0.8),  # 增强模糊强度
#         # step 3
#         Lambda(image=nonuniformity_optical, p=0.8, intensity=intensity),
#         Lambda(image=stripe_noise, p=0.8, intensity=intensity),
#         GaussNoise(std_range=(10.0 /255.0, 50.0 /255.0), p=0.8),  # 提高噪声强度
#     ], p=p)


# def deg_single(p=1, **params):
#     r = random.random()
#     if r < 0.33:
#         return Noise(p=p, **params)
#     elif r < 0.66:
#         return Blur(p=p, **params)
#     else:
#         return LC(p=p, **params)

# def deg_simple_step(patch_size, img, label):
#     img_list = {'image': np.array(img, dtype=np.uint8), 'mask': np.array(label, dtype=np.uint8)}
#     img_list = RandomCrop(patch_size[0], patch_size[1], always_apply=True)(**img_list)
#     img_list1 = LC()(**img_list)
#     img_list2 = Blur()(**img_list)
#     img_list3 = Noise()(**img_list)
#     return [img_list1, img_list2, img_list3]


# def deg_simple_chain(patch_size, img, label):
#     img_list = {'image': np.array(img, dtype=np.uint8), 'mask': np.array(label, dtype=np.uint8)}
#     img_list = RandomCrop(patch_size[0], patch_size[1], always_apply=True)(**img_list)
#     img_list1 = LC()(**img_list)
#     img_list2 = Blur()(**img_list1)
#     img_list3 = Noise()(**img_list2)
#     return [img_list1, img_list2, img_list3]


def aug_RC(patch_size, p=1):
    return RandomCrop(patch_size[0], patch_size[1], always_apply=True)


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size):
        super(TrainSetLoader).__init__()
        self.dataset_label_dir = dataset_dir + '/' + dataset_name
        if not isinstance(patch_size, Tuple):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        with open(self.dataset_label_dir + '/train.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        self.train_list = self.train_list
        self.transform = augumentation()

    def __getitem__(self, idx):
        label = Image.open((self.dataset_label_dir + '/imgs/' + self.train_list[idx]).replace('//', '/')).convert('L')
        img = Image.open((self.dataset_label_dir + '/imgs/' + self.train_list[idx]).replace('//', '/')).convert('L')

        if random.random() < 0.5:
            img_lists = deg_simple_step(self.patch_size, img, label)
            m = 0
        else:
            img_lists = deg_simple_chain(self.patch_size, img, label)
            m = 1

        cor_img_1 = Image.fromarray(img_lists[0]['image']).convert('L')
        cor_img_2 = Image.fromarray(img_lists[1]['image']).convert('L')
        cor_img_3 = Image.fromarray(img_lists[2]['image']).convert('L')
        label = Image.fromarray(img_lists[0]['mask']).convert('L')

        img_list = [np.array(cor_img_1, dtype=np.float32) / 255.0, np.array(cor_img_2, dtype=np.float32) / 255.0,
                    np.array(cor_img_3, dtype=np.float32) / 255.0]

        label = np.array(label, dtype=np.float32) / 255.0
        if len(label.shape) > 3:
            label = label[:, :, 0]

        img_list_patch, label_patch = self.transform(img_list, label)
        img_list_patch = toTensor(img_list_patch)
        label_patch = label_patch[np.newaxis, :]
        label_patch = torch.from_numpy(np.ascontiguousarray(label_patch))
        return img_list_patch, label_patch, m

    def __len__(self):
        return len(self.train_list)


def toTensor(img_list):
    for idx in range(len(img_list)):
        img_list[idx] = torch.from_numpy(np.ascontiguousarray(img_list[idx][np.newaxis, :]))
    img_list = torch.stack(img_list)
    return img_list


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, scenario, sub="hard"):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_label_dir = dataset_dir + '/' + dataset_name
        self.dataset_cor_dir = dataset_dir + '/' + dataset_name + '/' + scenario + '/' + sub
        with open(self.dataset_label_dir + '/test.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        self.test_list = self.test_list

    def __getitem__(self, idx):
        label = Image.open((self.dataset_label_dir + '/imgs/' + self.test_list[idx]).replace('//', '/')).convert('L')
        img = Image.open((self.dataset_cor_dir + '/' + self.test_list[idx]).replace('//', '/')).convert('L')
        img_list = [np.array(img, dtype=np.float32) / 255.0]
        label = np.array(label, dtype=np.float32) / 255.0
        h, w = label.shape

        if len(label.shape) > 3:
            label = label[:, :, 0]

        img_list = toTensor(img_list)
        label = label[np.newaxis, :]
        label = torch.from_numpy(np.ascontiguousarray(label))
        return img_list[0], label, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class InferSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.dataset_label_dir = dataset_dir + '/' + dataset_name
        self.test_list = os.listdir(self.dataset_label_dir + '/imgs/')

    def __getitem__(self, idx):
        image = Image.open((self.dataset_label_dir + '/imgs/' + self.test_list[idx]).replace('//', '/')).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0
        h, w = image.shape

        if len(image.shape) > 3:
            image = image[:, :, 0]

        image = PadImg(image, 32)

        image = image[np.newaxis, :]
        image = torch.from_numpy(np.ascontiguousarray(image))

        return image, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class VisSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.test_list = os.listdir(self.dataset_dir)

    def __getitem__(self, idx):
        image = Image.open((self.dataset_dir + '/' + self.test_list[idx]).replace('//', '/')).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0
        h, w = image.shape

        if len(image.shape) > 3:
            image = image[:, :, 0]

        image = PadImg(image, 32)

        image = image[np.newaxis, :]
        image = torch.from_numpy(np.ascontiguousarray(image))

        return image, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class augumentation(object):
    def __call__(self, inputs, target):
        if not isinstance(inputs, List):
            inputs = [inputs]

        if random.random() < 0.5:
            for idx in range(len(inputs)):
                inputs[idx] = inputs[idx][::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            for idx in range(len(inputs)):
                inputs[idx] = inputs[idx][:, ::-1]
            target = target[:, ::-1]
        return inputs, target


def random_crop(img_list, mask, patch_size):
    if not isinstance(img_list, List):
        img_list = [img_list]

    h, w = mask.shape
    base_h, base_w = patch_size
    if h < base_h or w < base_w:
        for idx in range(len(img_list)):
            img_list[idx] = np.pad(img_list[idx], ((0, max(h, base_h) - h), (0, max(w, base_w) - w)),
                                   mode='constant')
        mask = np.pad(mask, ((0, max(h, base_h) - h), (0, max(w, base_w) - w)), mode='constant')
        h, w = mask.shape

    h_start = random.randint(0, h - base_h)
    h_end = h_start + base_h
    w_start = random.randint(0, w - base_w)
    w_end = w_start + base_w

    for idx in range(len(img_list)):
        img_list[idx] = img_list[idx][h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_list, mask_patch


def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img
