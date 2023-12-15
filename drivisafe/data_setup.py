from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision.datasets.video_utils import VideoClips
import argparse


def get_dataloaders(
        dataset_path: Path,
        batch_size: int,
        num_workers: int,
        frame_rate: int,
        clip_length_in_frames: int,
        frames_between_clips: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    video_paths = [str(f / "video_garmin.avi") for f in dataset_path.iterdir() if f.is_dir()]
    video_clips = VideoClips(
        video_paths = [video_paths[:10]], 
        clip_length_in_frames = clip_length_in_frames, 
        frames_between_clips = frames_between_clips,
        num_workers = num_workers
    )
    dataloader = DataLoader(video_clips, batch_size = batch_size, shuffle = True)
    return
    # Iterate through the data loader
    for batch in dataloader:
        print(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_lab_size", type = float, default = 0.1)
    parser.add_argument("--test_size", type = float, default = 0.9)
    parser.add_argument("--train_unlab_size", type = float, default = 0.8)
    parser.add_argument("--val_size", type = float, default = 0.2)