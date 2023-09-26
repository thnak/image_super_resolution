import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torchvision.transforms import InterpolationMode, Lambda, Compose
import torchvision.transforms.functional as T
from torchvision.io import read_image, VideoReader, ImageReadMode
import json
from pathlib import Path
from time import sleep
from tqdm import tqdm

from utils.general import ground_up, convert_image_to_jpg


class Random_low_rs(Module):

    def __init__(self, input_shape, scale_factor):
        super(Random_low_rs, self).__init__()
        self.scale_factor = scale_factor
        self.shape = input_shape // scale_factor

    def forward(self, inputs):
        if random.random() != 1:
            return inputs[:, ::self.scale_factor, ::self.scale_factor]
        else:
            return T.resize(inputs, [self.shape, self.shapel])


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
        inputs = inputs.to(dtype=self.max_pixel_value.dtype)
        inputs /= self.max_pixel_value
        inputs -= self.mean
        inputs /= self.std
        return inputs


class DeNormalize(Module):
    """de-normalize tensor to range [0, 1]"""

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

    def forward(self, inputs):
        inputs *= self.std
        inputs += self.mean
        return inputs


class PIL_to_tanh(Module):
    def __init__(self, max_pixel_value=255.):
        super().__init__()
        self.register_buffer("max_pixel_value", torch.tensor(max_pixel_value))

    def forward(self, inputs):
        inputs = inputs.to(dtype=self.max_pixel_value.dtype)
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


class RandomNoisyImage(Module):
    def __init__(self, max_pixel_value=255.):
        super().__init__()
        from skimage.util import random_noise
        self.random_noise = random_noise
        self.max_pixel_value = max_pixel_value

    def forward(self, inputs):
        val = random.random()
        if val > 0.:
            val = val / 250
        noisy = self.random_noise(inputs.numpy(), var=val) * self.max_pixel_value
        noisy = np.clip(noisy, 0., 255).astype(np.uint8)
        return torch.from_numpy(noisy)


class Ychannel(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("image_weight", torch.tensor([65.481, 128.553, 24.966]))

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.to(device=self.image_weight.device)
        return torch.matmul(255. * inputs.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], self.image_weight) / 255. + 16.


class ColorJitter(Module):
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

    def __init__(self, json_path, target_size, scales_factor, calculateNorm=True, prefix=""):
        json_path = json_path if isinstance(json_path, Path) else Path(json_path)
        self.json_path = json_path
        with open(json_path.as_posix(), "r") as fi:
            self.samples = json.load(fi)
        target_size = ground_up(target_size, scales_factor)
        print(f"{prefix}{len(self.samples)} images with target shape {target_size} with scale factor {scales_factor}.")
        self.target_size = target_size
        self.scale_factor = scales_factor
        self.ran_position = Random_position(target_size=target_size)
        if calculateNorm:
            self.calculateNormValues()
        self.transform_lr = Compose([Random_low_rs(target_size, scales_factor),
                                     Normalize(mean=self.mean, std=self.std)])
        self.transform_hr = PIL_to_tanh()  # - > tanh

    def calculateNormValues(self):
        prefix = ""
        total_sample = len(self.samples)
        pbar = tqdm(range(total_sample), total=total_sample)
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])
        count = 0
        corrupt = []
        for x in pbar:
            try:
                try:
                    image = read_image(self.samples[x], ImageReadMode.RGB)  # CHW
                except Exception as ex:
                    self.samples[x] = convert_image_to_jpg(self.samples[x]).as_posix()
                    image = read_image(self.samples[x], ImageReadMode.RGB)  # CHW
            except:
                corrupt.append(x)
                continue
            count += image.size(1) * image.size(2)
            image = image.float()
            image /= 255.
            image = torch.unsqueeze(image, 0)
            psum += image.sum(axis=[0, 2, 3])
            psum_sq += (image ** 2).sum(axis=[0, 2, 3])

            pbar.set_description(f"{prefix}Collecting data to calculate mean, std...")
            # mean and std
            total_mean = psum / count
            total_var = (psum_sq / count) - (total_mean ** 2)
            total_std = torch.sqrt(total_var)
            self.mean = total_mean.cpu().numpy().tolist()
            self.std = total_std.cpu().numpy().tolist()
            pbar.set_description(f"{prefix}Using mean: {self.mean}, std: {self.std} for this dataset.")
        if len(corrupt) > 0:
            print(f"total {len(corrupt)} corrupt")
        for x in corrupt:
            print(f"removed corrupt image {self.samples[x]}")
            self.samples.pop(x)

    def set_transform_hr(self):
        """set transform for srgen"""
        self.transform_hr = Normalize(self.mean, self.std)
        return self

    def __getitem__(self, item):
        try:
            try:
                image = read_image(self.samples[item], ImageReadMode.RGB)  # CHW
            except Exception as ex:
                print(f"error {ex}")
                converted_image = convert_image_to_jpg(self.samples[item]).as_posix()
                sleep(0.1)
                self.samples[item] = converted_image
                with open(self.json_path.as_posix(), "w") as fi:
                    fi.write(json.dumps(self.samples))
                image = read_image(converted_image, ImageReadMode.RGB)  # CHW
        except:
            image = read_image(self.samples[item + 1], ImageReadMode.RGB)  # CHW

        image = self.ran_position(image)
        return self.transform_hr(image), self.transform_lr(image)

    def __len__(self):
        return len(self.samples)


class Noisy_dataset(Dataset):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, json_path, target_size, prefix=""):
        self.prefix = prefix
        json_path = json_path if isinstance(json_path, Path) else Path(json_path)
        with open(json_path.as_posix(), "r") as fi:
            self.samples = json.load(fi)

        self.target_size = target_size
        self.ran_position = Random_position(target_size=target_size)
        self.transform_lr = Compose([RandomNoisyImage(), Normalize(self.mean, self.std)])
        self.transform_hr = PIL_to_tanh()

    def __getitem__(self, item):
        image = read_image(self.samples[item], ImageReadMode.RGB)  # CHW
        image = self.ran_position(image)
        return self.transform_hr(image), self.transform_lr(image)

    def __len__(self):
        return len(self.samples)

    def calculateMeanSTD(self):
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])
        pbar = tqdm(self.samples, total=len(self.samples))
        count = 0
        detect_err = 0
        for i, di in enumerate(pbar):
            try:
                img = read_image(di, ImageReadMode.RGB).float()
            except:
                di = convert_image_to_jpg(di).as_posix()
                img = read_image(di, ImageReadMode.RGB).float()
                self.samples[i] = di
                detect_err += 1
            img /= 255.
            count += img.size(1) * img.size(2)
            img = torch.unsqueeze(img, dim=0)
            psum += img.sum(dim=[0, 2, 3])
            psum_sq += (img ** 2).sum(dim=[0, 2, 3])

            pbar.set_description(f"{self.prefix}Collecting data to calculate mean, std...")
            if i == len(self.samples) - 1:
                total_mean = psum / count
                total_var = (psum_sq / count) - total_mean ** 2
                total_std = torch.sqrt(total_var)
                self.mean = total_mean.cpu().numpy().tolist()
                self.std = total_std.cpu().numpy().tolist()
                pbar.set_description(f"{self.prefix}Using mean: {self.mean}, std: {self.std} for this dataset.")
        return detect_err


def init_dataloader(dataset, batch_size=16, shuffle=True, num_worker=2):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
                              pin_memory=True)  # note that we're passing the collate function here
    return train_loader, dataset


class dataset_for_inference(Dataset):
    def __init__(self, src):
        if isinstance(src, str):
            src = Path(src)
        video_reader = VideoReader(src.as_posix(), "video")
        video_reader.set_current_stream("video")
        meta_data = video_reader.get_metadata()
        fps = meta_data['video']['fps']

        fps = fps[0] if isinstance(fps, list) else fps
        total_frame = fps * meta_data['video']['duration'][0]
        self.total_frame = int(total_frame)
        self.fps = fps
        self.video_reader = video_reader
        frame = next(video_reader)['data']
        self.shape = frame.shape  # CHW
        video_reader.seek(0, keyframes_only=True)

    def __len__(self):
        return self.total_frame

    def __getitem__(self, item):
        frame = next(self.video_reader)['data']
        return frame


def init_dataloader_for_inference(src, worker, batch_size):
    if isinstance(src, str):
        src = Path(src)
    dataset = dataset_for_inference(src)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=worker)
    return dataloader, dataset
