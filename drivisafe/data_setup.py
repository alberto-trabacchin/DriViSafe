from pathlib import Path
import torch
from torchvision.datasets.video_utils import VideoClips
from torch.utils.data import DataLoader
from typing import Tuple
import os


def get_dataloaders(
        dataset_path: Path,
        batch_size: int,
        num_workers: int,
        frame_rate: int,
        clip_length_in_frames: int,
        frames_between_clips: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    video_paths = [str(f / "video_garmin.avi") for f in dataset_path.iterdir() if f.is_dir()]
    print(video_paths[:10])
    exit()
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
    get_dataloaders(
        dataset_path = Path("/mnt/d/Datasets/Dr(eye)ve/DREYEVE_DATA/"),
        batch_size = 32,
        num_workers = os.cpu_count(),
        frame_rate = 25,
        clip_length_in_frames = 7500
    )