import json
import math
from pathlib import Path

from PIL import Image
from torch import nn

ACT_LIST = (nn.LeakyReLU, nn.Hardswish, nn.ReLU, nn.ReLU6,
            nn.SiLU, nn.Tanh, nn.Sigmoid, nn.ELU, nn.PReLU,
            nn.Softmax, nn.Hardsigmoid, nn.GELU, nn.Softsign, nn.Softplus)
IMG_FORMATS = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo', '.pfm', '.jpg', '.jpeg',
               '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']  # acceptable image suffixes
VID_FORMATS = ['.asf', '.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv',
               '.gif']


def ground_up(intput_number, stride):
    if intput_number % stride == 0:
        return intput_number
    else:
        return math.ceil(intput_number / stride) * stride


def fix_problem_with_reuse_activation_funtion(act):
    """to fix problem with thop when reuse activation funtion"""
    if isinstance(act, bool):
        return act
    act = str(act)
    act = f"nn.{act}" if "nn." not in act else act
    if not ("(" in act and ")" in act):
        act = f"{act}()"
    act = eval(act)
    return act


def autopad(kernel_size: int | tuple[int] | list[int], pad_size=None, dilation=1):
    """Pad to 'same' shape outputs"""
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [dilation * (x - 1) + 1 for
                                                                                             x in
                                                                                             kernel_size]  # actual kernel-size
    if pad_size is None:
        pad_size = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]  # auto-pad
    return pad_size


def create_data_lists(train_folders, test_folders, min_size, output_folder="./"):
    """
    Create lists for images in the training set and each of the test sets.

    :param train_folders: folders containing the training images; these will be merged
    :param test_folders: folders containing the test images; each test folder will form its own test set
    :param min_size: minimum width and height of images to be considered
    :param output_folder: save data lists here
    """
    print("\nCreating data lists... this may take some time.\n")
    train_images = []
    for d in train_folders:
        for i in Path(d).iterdir():
            if i.suffix in IMG_FORMATS:
                image = Image.open(i.as_posix())
                if image.width < min_size or image.height < min_size:
                    print(f"ignore small image {i.as_posix()} require {min_size}")
                else:
                    train_images.append(i.as_posix())

    print("There are %d images in the training data.\n" % len(train_images))
    save_dir = Path(output_folder)
    save_dir = save_dir / 'train_images.json'
    with open(save_dir.as_posix(), 'w') as j:
        json.dump(train_images, j)
    train_images = []
    for d in train_folders:
        for i in Path(d).iterdir():
            if i.suffix in IMG_FORMATS:
                image = Image.open(i.as_posix())
                if image.width < min_size or image.height < min_size:
                    print(f"ignore small image {i.as_posix()} require {min_size}")
                else:
                    train_images.append(i.as_posix())

    print("There are %d images in the validating data.\n" % len(train_images))
    save_dir = Path(output_folder)
    save_dir = save_dir / 'train_images.json'
    with open(save_dir.as_posix(), 'w') as j:
        json.dump(train_images, j)

    print("JSONS containing lists of Train and Test images have been saved to %s\n" % save_dir.as_posix())


def convert_image_to_jpg(image_file: str | Path):
    if isinstance(image_file, str):
        image_file = Path(image_file)
    image = Image.open(image_file.as_posix())
    save_dir = image_file.with_suffix(".jpg")
    image.save(save_dir)
    return save_dir



def intersect_dicts(da, db, exclude=()):
    """Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values"""
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}
