import os
import argparse
from pathlib import Path
import logging
import shutil
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", 
    datefmt='%Y-%m-%d,%H:%M:%S', 
    level=logging.INFO
)
logger = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, required=True)
parser.add_argument("--save-path", type=str, default="./data/")
parser.add_argument("--fps", type=int, default=3)
parser.add_argument("--resize", type=int, nargs=2, default=None)
args = parser.parse_args()


def vid2img(video_path, save_path, resize, fps):
    vid_id = video_path.parent.stem
    logger.info(f"Converting {video_path} to images...")
    if resize is None:
        os.system(f'ffmpeg -i {str(video_path)} -vf fps={fps} {str(save_path / vid_id)}_%d.jpg')
    else:
        os.system(f'ffmpeg -i {str(video_path)} -vf "scale={resize[0]}:{resize[1]}, fps={fps}" {str(save_path / vid_id)}_%d.jpg')
    return True


if __name__ == "__main__":
    dataset_path = Path(args.dataset_path).absolute()
    video_paths = [f for f in (dataset_path / "DREYEVE_DATA").iterdir() if f.is_dir()]
    video_paths = [f / "video_garmin.avi" for f in video_paths]
    save_path = Path(args.save_path) / "dreyeve" / "images"
    if not save_path.exists():
        save_path.mkdir(parents=True)
    else:
         shutil.rmtree(save_path)
         save_path.mkdir(parents=True)
    for p in tqdm(video_paths):
            vid2img(p, save_path, args.resize, args.fps)
    