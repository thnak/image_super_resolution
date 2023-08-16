import argparse
import torch
from torchvision.io import read_image, ImageReadMode, VideoReader
from pathlib import Path
from utils.general import VID_FORMATS
from utils.datasets import Normalize, RGB2BGR
from utils.models import Tanh_to_PIL, Tanh_to_ImageArray
from utils.ffmpeg import FFMPEG_recorder
from tqdm import tqdm


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
        tanh_2_pil = Tanh_to_PIL()
        image = read_image(src.as_posix(), ImageReadMode.RGB)
        frame = norm_(image).unsqueeze(0).to(device)
        frames = tanh_2_pil(model(frame))

        for frame in frames:
            frame.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--src", type=str, default="")
    parser.add_argument("--save_dir", type=str)
    opt = parser.parse_args()
    runer(model=opt.model, src=opt.src, save_dir=opt.save_dir)

