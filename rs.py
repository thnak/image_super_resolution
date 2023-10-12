from __future__ import annotations

import argparse
import torch
from torch import autocast
from torchvision.io import read_image, ImageReadMode, VideoReader, write_png
from torchvision.transforms.functional import resize, InterpolationMode
from pathlib import Path
from utils.general import VID_FORMATS, convert_image_to_jpg
from utils.datasets import Normalize, RGB2BGR, init_dataloader_for_inference
from utils.models import Tanh2PIL, TanhToArrayImage
from utils.ffmpeg import FFMPEG_recorder
from tqdm import tqdm


def sliding_window(image: torch.Tensor, step: int | list[int, int] | tuple[int, int], windowSize=None):
    """accept ...CHW"""
    if windowSize is None:
        windowSize = step
    if isinstance(step, int):
        step = [step] * 2
    step[0] = min(image.shape[-2], step[0])
    step[1] = min(image.shape[-1], step[1])

    for y in range(0, image.shape[-2], step[0]):
        for x in range(0, image.shape[-1], step[1]):
            yield step, x, y, image[..., y:y + windowSize, x:x + windowSize]


def total_inter_sliding_window(image: torch.Tensor, step: int):
    n_dims = image.dim()
    if n_dims == 3:
        _, h, w = image.size()
        return len([_ for _ in range(0, h, step)]) * len([_ for _ in range(0, w, step)])
    else:
        return 0


def runer(**kwargs):
    model_dir = Path(kwargs["model"])
    src = Path(kwargs['src'])
    result_dir = Path(kwargs['save_dir'])
    step_size = kwargs['window_size']
    batch_size = kwargs['batch_size']
    worker = kwargs['worker']
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = torch.jit.load(model_dir, "cpu")
    model.to(device).eval()
    # model = torch.jit.optimize_for_inference(model)
    norm_ = Normalize().to(device)
    rgb2bgr = RGB2BGR().to(device)

    if src.suffix in VID_FORMATS:
        tanh_2_pil = TanhToArrayImage().to(device)
        video_writer = None
        result_dir = result_dir.with_suffix('.mp4')
        dataloader, dataset = init_dataloader_for_inference(src, 0, batch_size=batch_size)
        pbar = tqdm(dataloader, total=len(dataloader))
        for idx, frames in enumerate(pbar):
            frames = frames.to(device).float()
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                frames = norm_(frames)
                sr = model(frames)
            sr = tanh_2_pil(sr)
            sr = rgb2bgr(sr)
            if video_writer is None:
                video_writer = FFMPEG_recorder(save_path=result_dir.as_posix(),
                                               videoDimensions=(sr.size(-1), sr.size(-2)), fps=dataset.fps)
            for frame in sr:
                frame = frame.cpu()
                frame = frame.permute([1, 2, 0]).numpy()
                pbar.desc = f"{frame.shape} {frame.dtype}"
                video_writer.writeFrame(frame)
        video_writer.stopRecorder()
        video_writer.addAudio(src.as_posix())

    else:
        tanh_2_pil = TanhToArrayImage().to(device)
        if src.suffix not in ['.jpg', ".png"]:
            src = convert_image_to_jpg(src)
        image = read_image(src.as_posix(), ImageReadMode.RGB)
        c, h, w = image.size()
        print("input shape", image.size())
        # image2 = resize(image, [h * 4, w * 4], InterpolationMode.BICUBIC, antialias=True)
        # write_jpeg(image2, "bicubic.jpg")
        result_image = None
        image_width = 0
        high, width = 0, 0
        pbar = tqdm(sliding_window(image, step_size), total=total_inter_sliding_window(image, step_size))
        for step_size, _, _, window_img in pbar:
            window_img = window_img.unsqueeze(0).to(device)
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                window_img = model(window_img)
            # window_img = tanh_2_pil(window_img)
            r_b, r_c, r_h, r_w = window_img.size()
            if result_image is None:
                if step_size[0] == r_h:
                    result_image = torch.zeros_like(image)
                else:
                    scale_factor = r_h / step_size[0]
                    result_image = torch.zeros([r_c, int(h * scale_factor), int(w * scale_factor)], dtype=torch.uint8)
                _, image_high, image_width = result_image.size()

            for frame in window_img:
                c, h, w = frame.size()
                result_image[..., high:high + h, width:width + w] = frame.cpu()
                width += w
                if width >= image_width:
                    high += h
                    width = 0
        result_dir = result_dir.with_suffix(".png")
        write_png(result_image, result_dir.as_posix())
        print("output shape", result_image.shape, f"{result_dir.as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--src", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="result.jpg")
    parser.add_argument("--window_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--worker", type=int, default=4)
    opt = parser.parse_args()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # torch.jit.enable_onednn_fusion(True)

    runer(**opt.__dict__)
