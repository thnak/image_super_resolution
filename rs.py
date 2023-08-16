import argparse
import torch
from torchvision.io import read_image, ImageReadMode, VideoReader, write_jpeg
from torchvision.transforms.functional import resize, InterpolationMode
from pathlib import Path
from utils.general import VID_FORMATS
from utils.datasets import Normalize, RGB2BGR
from utils.models import Tanh_to_PIL, Tanh_to_ImageArray
from utils.ffmpeg import FFMPEG_recorder
from tqdm import tqdm
from PIL import Image


def sliding_window(image, stepSize, windowSize=None):
    if windowSize is None:
        windowSize = stepSize
    for y in range(0, image.shape[1], stepSize):
        for x in range(0, image.shape[2], stepSize):
            yield (x, y, image[..., y:y + windowSize, x:x + windowSize])


def runer(**kwargs):
    model_dir = Path(kwargs["model"])
    src = Path(kwargs['src'])
    result_dir = Path(kwargs['save_dir'])
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_dir)
    model.to(device).eval()
    norm_ = Normalize()
    rgb2bgr = RGB2BGR()
    if src.suffix in VID_FORMATS:
        video_writer = None
        video = VideoReader(src.as_posix(), "video")
        video.set_current_stream("video")
        metadata = video.get_metadata()
        fps = metadata['video']['fps']
        tanh_2_pil = Tanh_to_ImageArray()

        fps = fps[0] if isinstance(fps, list) else fps
        total_frame = fps * metadata['video']['duration'][0]
        pbar = tqdm(video, total=int(total_frame))
        for idx, data_dict in enumerate(pbar):
            frame = data_dict['data'].to(device)
            frame = norm_(frame)
            frame = frame.unsqueeze(0)
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
    else:
        tanh_2_pil = Tanh_to_ImageArray()
        image = read_image(src.as_posix(), ImageReadMode.RGB)
        c, h, w = image.size()
        print("input shape", image.size())
        image2 = resize(image, [h * 4, w * 4], InterpolationMode.BICUBIC, antialias=True)
        write_jpeg(image2, "bicubic.jpg")
        result_image = torch.zeros([c, h * 4, w * 4], device=device, dtype=torch.uint8)
        c, image_high, image_width = result_image.size()

        high, width = 0, 0
        for _, _, window_img in sliding_window(image, 96):
            window_img = norm_(window_img).unsqueeze(0).to(device)
            window_img = tanh_2_pil(model(window_img))
            for frame in window_img:
                c, h, w = frame.size()
                result_image[..., high:high + h, width:width + w] = frame
                width += w
                if width >= image_width:
                    high += h
                    width = 0

        write_jpeg(result_image, 'result.jpg')
        print(result_image.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--src", type=str, default="")
    parser.add_argument("--save_dir", type=str)
    opt = parser.parse_args()
    runer(model=opt.model, src=opt.src, save_dir=opt.save_dir)
