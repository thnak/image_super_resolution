import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torchvision.transforms import Compose, InterpolationMode, Lambda
import torchvision.transforms.functional as T
from torchvision.io import read_image, ImageReadMode
import json
import numpy as np
from pathlib import Path
from general import ground_up


class Random_low_rs(Module):
    BILINEAR = InterpolationMode.BILINEAR
    BICUBIC = InterpolationMode.BICUBIC

    def __init__(self, input_shape, scale_factor):
        super(Random_low_rs, self).__init__()
        self.scale_factor = scale_factor
        self.input_shape = input_shape

    def forward(self, inputs):
        size = self.input_shape / self.scale_factor
        if random.random() < 1 / 3:
            inputs = inputs[:, ::self.scale_factor, ::self.scale_factor]
        else:
            inputs = T.resize(inputs, [size] * 2, self.BILINEAR if random.random() < 0.5 else self.BICUBIC)
        return inputs


class Random_position(Module):
    def __init__(self, target_size):
        super(Random_position, self).__init__()
        self.target_size = target_size

    def forward(self, inputs: torch.Tensor):
        c, h, w = inputs.size()
        left = random.randint(1, w - self.target_size)
        bottom = random.randint(1, h - self.target_size)
        right = left + self.target_size
        top = bottom + self.target_size
        inputs = inputs[:, bottom:top, left:right]
        return inputs


class Normalize(Module):
    """input int tensor in NCHW or CHW format and return the same format"""

    def __init__(self,
                 mean: tuple[float, float, float] | tuple[float] | list[float, float, float] | list[float],
                 std: tuple[float, float, float] | tuple[float] | list[float, float, float] | list[float],
                 pixel_max_value=255.):
        super(Normalize, self).__init__()
        self.pixel_max_value = pixel_max_value
        self.mean = mean
        self.std = std

    def forward(self, inputs: torch.Tensor):
        n_dims = inputs.dim()
        inputs = inputs.float()
        inputs /= self.pixel_max_value
        if n_dims == 4:
            n_dept, c, h, w = inputs.shape
            assert c == len(self.mean), f"len of self.mean ({len(self.mean)}) must be equal to image channel ({c})"
            for n in range(n_dept):
                for x in range(c):
                    inputs[n, x, ...] = (inputs[n, x, ...] - self.mean[x]) / self.std[x]
        elif n_dims == 3:
            c, h, w = inputs.shape
            assert c == len(self.mean), f"len of self.mean ({len(self.mean)}) must be equal to image channel ({c})"
            for x in range(c):
                inputs[x, ...] = (inputs[x, ...] - self.mean[x]) / self.std[x]
        else:
            raise f"inputs tensor must be 3 or 4 dimension, got {n_dims}"
        return inputs


class Decode_tensor_from_predict(Module):
    def __init__(self):
        super().__init__()
        self.source = None

    def forward(self, inputs: torch.Tensor):
        if self.source is None:
            self.source = -1 if torch.min(inputs) < 0.0 else 0
        if self.source == -1:
            inputs = 1 + inputs
            inputs /= 2
        else:
            inputs *= 255.
        return inputs.round().numpy()


class ColorJiter(Module):
    def __init__(self,
                 brightness=0.2,
                 contrast=0.2, saturation=0.2, hue=0.2,
                 always_apply=False, p=0.5):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.allways_apply = always_apply
        self.p = p

    def forward(self, inputs: torch.Tensor):
        """
        value = brightness = contrast = saturation = 1 is origin image
        0 < value < 1 is lower
        value > 1 is higher
        for hue must in range [-0.5, 0.5]

        always_apply=True use this augment for all the time
        """
        brightness = max(self.brightness, 0)
        contrast = max(self.contrast, 0)
        saturation = max(self.saturation, 0)
        hue = min(self.hue, 0.5)
        hue = max(hue, 0)

        assert 0 <= self.p <= 1, f'Probability must in range [0, 1]'

        if random.random() <= self.p or self.allways_apply:
            brightness = random.uniform(1 - brightness, 1 + brightness)
            if brightness != 1:
                inputs = T.adjust_brightness(inputs, brightness)

            contrast = random.uniform(1 - contrast, 1 + contrast)
            if contrast != 1:
                inputs = T.adjust_contrast(inputs, contrast)

            saturation = random.uniform(1 - saturation, 1 + saturation)
            if saturation != 1:
                inputs = T.adjust_saturation(inputs, saturation)

            hue = random.uniform(-hue, hue)
            if hue != 0:
                inputs = T.adjust_hue(inputs, hue)
        return inputs


class SR_dataset(Dataset):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, json_path, target_size, scales_factor, prefix=""):
        json_path = json_path if isinstance(json_path, Path) else Path(json_path)
        with open(json_path.as_posix(), "r") as fi:
            self.samples = json.load(fi)
        target_size = ground_up(target_size, scales_factor)
        print(f"{prefix}{len(self.samples)} images with target shape {target_size} with scale factor {scales_factor}.")
        self.target_size = target_size
        self.scale_factor = scales_factor
        self.crop = Random_position(target_size=target_size)
        self.transform_lr = Normalize(mean=self.mean, std=self.std)
        self.transform_hr = Lambda(lambd=lambda x: 2. * (x / 255.) - 1)

    def __getitem__(self, item):
        image = read_image(self.samples[item], ImageReadMode.RGB)  # CHW
        image = self.crop(image)
        high_rs_image = self.transform_hr(image)
        low_rs_image = image[:, ::self.scale_factor, ::self.scale_factor]
        low_rs_image = self.transform_lr(low_rs_image)
        return high_rs_image, low_rs_image

    def __len__(self):
        return len(self.samples)


def init_dataloader(dataset, batch_size=16, shuffle=True, num_worker=2):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
                              pin_memory=True)  # note that we're passing the collate function here
    return train_loader, dataset
