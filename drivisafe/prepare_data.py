from pathlib import Path
from torchvision.io import VideoReader, write_jpeg
import os
import multiprocessing as mp
import json
import argparse
import splitfolders
from typing import Tuple


def worker(args) -> bool:
    """
    Convert a video into a series of JPEG images.

    Parameters:
    args (tuple): A tuple containing the video path, save path, quality of the images, 
                  maximum number of frames to extract, and the sampling step.

    Returns:
    bool: True if the video was successfully converted, False otherwise.
    """
    video_path, save_path, quality, max_frames, sampling_step = args
    video_id = video_path.parent.stem
    reader = VideoReader(str(video_path), "video")
    print(f"Converting video {video_id}...")
    for count, frame in enumerate(reader):
        if (max_frames is not None) and ((count + 1) > sampling_step * max_frames):
            break
        elif (count % sampling_step) != 0:
            continue
        image = frame["data"]
        fpath = save_path / f"{video_id}_{count}.jpg"
        write_jpeg(image, str(fpath), quality)
    print(f"Video {video_id} converted.")
    return True


def create_labelstudio_json(save_path: Path) -> None:
    """
    Generate a JSON file for Label Studio with paths to the images that need to be labeled.

    Parameters:
    save_path (Path): The path where the JSON file will be saved.

    Returns:
    None
    """
    print("Generating local-storage.json...")
    frame_names = [f.name for f in save_path.iterdir() if f.is_file()]
    data = []
    for f in frame_names:
        lab_studio_frame_path = "/data/local-files?d=Dr(eye)ve/data_frames/" + str(f)
        data.append({"image": [str(lab_studio_frame_path)]})
    loc_storage_path = str(save_path / "local-storage.json")
    with open(loc_storage_path, "w") as f:
        json.dump(data, f)
    print("local-storage.json saved to " + loc_storage_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type = str, required = True, help = "Path to the dataset: /..../Dr(eye)ve")
    parser.add_argument("--sampling_step", type = int, default = 10)
    parser.add_argument("--max_frames", type = int, default = None)
    parser.add_argument("--quality", type = int, default = 50)
    parser.add_argument("--n_workers", type = int, default = os.cpu_count())
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path) / "DREYEVE_DATA"
    save_path = dataset_path / "data_frames"
    save_path.mkdir(exist_ok = True)

    # Get all video paths
    video_paths = [(f / "video_garmin.avi") for f in dataset_path.iterdir() if f.is_dir()]

    # Convert videos to frames (parallelized)
    p = mp.Pool(processes = args.n_workers)
    worker_args = [(
        video_path,
        save_path,
        args.quality,
        args.max_frames,
        args.sampling_step
    ) for video_path in video_paths]
    results = p.map(worker, worker_args)
    p.close()
    p.join()

    # Check that all videos were converted
    assert(len(results) == len(video_paths))
    assert(all(results))

    # Create local-storage.json to load frames into Label Studio
    create_labelstudio_json(save_path = dataset_path)