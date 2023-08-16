import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torchvision.transforms import InterpolationMode, Lambda, Compose
import torchvision.transforms.functional as T
from torchvision.io import read_image, ImageReadMode
import json
from pathlib import Path
from utils.general import ground_up


class Random_low_rs(Module):
    BILINEAR = InterpolationMode.BILINEAR
    BICUBIC = InterpolationMode.BICUBIC

    def __init__(self, input_shape, scale_factor):
        super(Random_low_rs, self).__init__()
        self.scale_factor = scale_factor
        self.input_shape = input_shape

    def forward(self, inputs):
        size = int(self.input_shape / self.scale_factor)
        random_value = random.random()
        if random_value < 1 / 3:
            inputs = inputs[:, ::self.scale_factor, ::self.scale_factor]
        else:
            inputs = T.resize(inputs, [size, size],
                              self.BILINEAR if random_value < 2 / 3 else self.BICUBIC,
                              antialias=False)
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
    """return value from torch Tensor PIL image to norm"""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255., dim=3):
        super().__init__()
        if dim == 4:
            mean = torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            std = torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            mean = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
            std = torch.tensor(std).unsqueeze(1).unsqueeze(2)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("max_pixel_value", torch.tensor(max_pixel_value))

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.float()
        inputs /= self.max_pixel_value
        inputs -= self.mean
        inputs /= self.std
        return inputs


class PIL_to_tanh(Module):
    def __init__(self, max_pixel_value=255.):
        super().__init__()
        self.register_buffer("max_pixel_value", torch.tensor(max_pixel_value))

    def forward(self, inputs):
        if inputs.dtype == torch.uint8:
            inputs = inputs.float()
        inputs /= self.max_pixel_value
        return 2. * inputs - 1


class RGB2BGR(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(inputs):
        dim = inputs.dim()
        if dim == 3:
            r, g, b = inputs[0, ...], inputs[1, ...], inputs[2, ...]
            inputs = torch.stack([b, g, r])
        elif dim == 4:
            r, g, b = inputs[..., 0, :, :], inputs[..., 1, :, :], inputs[..., 2, :, :]
            inputs = torch.stack([b, g, r], dim=1)
        else:
            raise f"not support {dim} channel input"
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
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, json_path, target_size, scales_factor, prefix=""):
        json_path = json_path if isinstance(json_path, Path) else Path(json_path)
        with open(json_path.as_posix(), "r") as fi:
            self.samples = json.load(fi)
        target_size = ground_up(target_size, scales_factor)
        print(f"{prefix}{len(self.samples)} images with target shape {target_size} with scale factor {scales_factor}.")
        self.target_size = target_size
        self.scale_factor = scales_factor
        self.crop = Random_position(target_size=target_size)
        self.transform_lr = Compose([Random_low_rs(target_size, scales_factor),
                                     Normalize(mean=self.mean, std=self.std)])
        self.transform_hr = PIL_to_tanh()  # - > tanh

    def set_transform_hr(self):
        """set transform for srgen"""
        self.transform_hr = Normalize(self.mean, self.std)

    def __getitem__(self, item):
        image = read_image(self.samples[item], ImageReadMode.RGB)  # CHW
        image = self.crop(image)
        return self.transform_hr(image), self.transform_lr(image)

    def __len__(self):
        return len(self.samples)


def init_dataloader(dataset, batch_size=16, shuffle=True, num_worker=2):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
                              pin_memory=True)  # note that we're passing the collate function here
    return train_loader, dataset
