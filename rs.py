import argparse
import torch
from torchvision.io import read_image, ImageReadMode, VideoReader, write_jpeg
from torchvision.transforms.functional import resize, InterpolationMode
from pathlib import Path
from utils.general import VID_FORMATS, convert_image_to_jpg
from utils.datasets import Normalize, RGB2BGR
from utils.models import Tanh_to_PIL, Tanh_to_ImageArray
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
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = torch.jit.load(model_dir, "cpu")
    model.to(device).eval()
    norm_ = Normalize().to(device)
    rgb2bgr = RGB2BGR().to(device)

    half = device.type == "cuda"
    if half:
        model.half()
    if src.suffix in VID_FORMATS:
        video_writer = None
        result_dir = result_dir.with_suffix('.mp4')
        video = VideoReader(src.as_posix(), "video")
        video.set_current_stream("video")
        metadata = video.get_metadata()
        fps = metadata['video']['fps']
        tanh_2_pil = Tanh_to_ImageArray().to(device)

        fps = fps[0] if isinstance(fps, list) else fps
        total_frame = fps * metadata['video']['duration'][0]
        pbar = tqdm(video, total=int(total_frame))
        for idx, data_dict in enumerate(pbar):
            frame = data_dict['data'].to(device)

            frame = norm_(frame)
            frame = frame.unsqueeze(0)
            if half:
                frame = frame.half()
            sr = tanh_2_pil(model(frame))
            sr = rgb2bgr(sr)
            if video_writer is None:
                video_writer = FFMPEG_recorder(save_path=result_dir.as_posix(),
                                               videoDimensions=(sr.size(-1), sr.size(-2)), fps=fps)
            for frame in sr:
                frame = frame.cpu()
                frame = frame.permute([1, 2, 0]).numpy()
                pbar.desc = f"{frame.shape} {frame.dtype}"
                video_writer.writeFrame(frame)
        video_writer.stopRecorder()
        video_writer.addAudio(src.as_posix())
    else:
        tanh_2_pil = Tanh_to_ImageArray().to(device)
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
            window_img = norm_(window_img).unsqueeze(0).to(device)
            if half:
                window_img = window_img.half()
            window_img = tanh_2_pil(model(window_img))
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

        write_jpeg(result_image, 'result.jpg')
        print("output shape", result_image.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--src", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="result.jpg")
    parser.add_argument("--window_size", type=int, default=96)
    opt = parser.parse_args()
    runer(model=opt.model, src=opt.src, save_dir=opt.save_dir, window_size=opt.window_size)
