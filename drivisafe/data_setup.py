from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
from typing import List, Dict
import json
import argparse


class DreyeveDataset(Dataset):
    def __init__(
            self,
            paths: List[Path],
            labels: List[int] = None,
            transform: transforms = None
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index: int) -> Tuple[str, int]:
        path = self.paths[index],
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image, None

    def __len__(self) -> int:
        return len(self.paths)


def convert_annot_path(
        root_path: Path,
        annot_path: Path
) -> Path:
    """
    Convert the annotation path to the full frame path.

    This function takes the root path and the relative frame annotation paths,
    and constructs the full frame path by joining the root path with the last two parts
    of the frame annotation paths.

    Args:
        root_path (Path): The root path.
        frame_annot_paths (Path): The frame annotation paths.

    Returns:
        Path: The full frame path.

    Example:
        root_path = /mnt/c/Dr(eye)ve/
        annot_path = /data/local-files?d=Dr(eye)ve/data_frames/04_4420.jpg
        returns: /mnt/c/Dr(eye)ve/data_frames/04_4420.jpg
    """
    parts = annot_path.parts
    full_path = root_path / parts[-2] / parts[-1]
    return full_path
    

def get_data_annotations(
        root_path: Path,
        annot_path: Path, 
        labels_to_idx: Dict[str, int]
) -> Tuple[List[Path], List[int]]:
    """
    Get the data annotations from JSON-format annotations made with Label Studio.

    Args:
        root_path (Path): The root path of the dataset: /.../Dr(eye)ve/.
        annot_path (Path): The annotation path of the JSON file.
        labels_to_idx (Dict[str, int]): The mapping from labels to indices.

    Returns:
        Tuple[List[Path], List[int]]: The absolute frame paths and respective labels.
    """
    data = json.load(annot_path.open(mode = "r"))
    labels = [labels_to_idx[d["choice"]] for d in data]
    annot_paths = [Path(d["image"][0]) for d in data]
    full_paths = [convert_annot_path(root_path, p) for p in annot_paths]
    return full_paths, labels


def make_datasets(
        root_path: Path,
        frames_path: Path,
        annot_path: Path,
        train_lab_size: float,
        test_size: float,
        train_unlab_size: float,
        val_size: float,
        labels_to_idx: Dict[str, int],
        shuffle: bool = True,
        seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    frame_paths = frames_path.iterdir()
    data, labels = get_data_annotations(root_path, annot_path, labels_to_idx)
    print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type = str, required = True, help = "Path to the dataset: /..../Dr(eye)ve")
    parser.add_argument("--train_lab_size", type = float, default = 0.1)
    parser.add_argument("--test_size", type = float, default = 0.9)
    parser.add_argument("--train_unlab_size", type = float, default = 0.8)
    parser.add_argument("--val_size", type = float, default = 0.2)
    args = parser.parse_args()

    labels_to_idx = {
        "Dangerous": 0,
        "NOT Dangerous": 1
    }
    make_datasets(
        root_path = Path(args.root_dir),
        frames_path = Path(args.root_dir) / "data_frames",
        annot_path = Path(args.root_dir) / "data_annotations.json",
        train_lab_size = args.train_lab_size,
        test_size = args.test_size,
        train_unlab_size = args.train_unlab_size,
        val_size = args.val_size,
        labels_to_idx = labels_to_idx,
        shuffle = True,
        seed = 42
    )