from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
from typing import List, Dict
import json
import random
import argparse
from matplotlib import pyplot as plt


class DreyeveDataset(Dataset):
    def __init__(
            self,
            data: List[Path],
            labels: List[int] = None,
            transform: transforms = None,
            labels_to_idx: Dict[str, int] = None
    ) -> None:
        self.data = data
        self.labels = labels
        self.transform = transform
        self.labels_to_idx = labels_to_idx
    
    def __getitem__(self, index: int) -> Tuple[str, int]:
        path = self.data[index]
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image, None

    def __len__(self) -> int:
        return len(self.data)
    
    def plot_sample(self, index: int) -> None:
        image, label = self[index]
        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0))
        if (self.labels_to_idx is not None) and (label is not None):
            keys = list(self.labels_to_idx.keys())
            values = list(self.labels_to_idx.values())
            label = keys[values.index(label)]
        ax.set_title(label)


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
    

def get_labeled_data(
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


def get_unlabeled_data(
        data_frames_path: Path,
        labeled_data: List[Path]
) -> List[Path]:
    """
    Get the unlabeled data from the data frames path.

    Args:
        data_frames_path (Path): The path to the data frames.
        labeled_data (List[Path]): The labeled data.

    Returns:
        List[Path]: The unlabeled data.
    """
    unlabeled_data = [p for p in data_frames_path.iterdir() if p not in labeled_data]
    return unlabeled_data


def split_data(
        lab_data: List[Path],
        unlab_data: List[Path],
        labels: List[int],
        train_lab_size: float,
        test_size: float,
        train_unlab_size: float,
        val_size: float,
        shuffle: bool,
        seed: int
) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[int], List[int]]:
    """
    Split the data into labeled and unlabeled training, testing and validation sets.

    Args:
        lab_data (List[Path]): The labeled data.
        unlab_data (List[Path]): The unlabeled data.
        labels (List[int]): The labels.
        train_lab_size (float): The size of the labeled training set.
        test_size (float): The size of the testing set.
        train_unlab_size (float): The size of the unlabeled training set.
        val_size (float): The size of the validation set.
        shuffle (bool): Whether to shuffle the data.
        seed (int): The random seed.

    Returns:
        Tuple[List[Path], List[Path], List[Path], List[Path], List[int], List[int]]: The training and testing sets.
    """
    assert(len(lab_data) >= 2)
    assert(train_lab_size + test_size <= 1)
    assert(train_unlab_size + val_size <= 1)

    # Shuffle the data
    if shuffle:
        lab_zip = list(zip(lab_data, labels))
        random.seed(seed)
        random.shuffle(lab_zip)
        random.shuffle(unlab_data)
        lab_data, labels = zip(*lab_zip)

    # Calculate the sizes of the sets
    train_lab_len = max(1, int(train_lab_size * len(lab_data)))
    test_len = max(1, int(test_size * len(lab_data)))
    train_unlab_len = max(1, int(train_unlab_size * len(unlab_data)))
    val_len = max(1, int(val_size * len(unlab_data)))
    train_lab_idx = train_lab_len
    test_idx = train_lab_idx + test_len
    train_unlab_idx = train_unlab_len
    val_idx = train_unlab_idx + val_len
    
    # Split the labeled data
    train_lab_data = lab_data[:train_lab_idx]
    test_data = lab_data[train_lab_idx : test_idx]
    train_labels = labels[:train_lab_idx]
    test_labels = labels[train_lab_idx : test_idx]

    # Split the unlabeled data
    train_unlab_data = unlab_data[:train_unlab_idx]
    valid_data = unlab_data[train_unlab_idx : val_idx]

    return train_lab_data, train_unlab_data, test_data, \
           valid_data, train_labels, test_labels


def make_datasets(
        root_path: Path,
        frames_path: Path,
        annot_path: Path,
        train_lab_size: float,
        test_size: float,
        train_unlab_size: float,
        val_size: float,
        labels_to_idx: Dict[str, int],
        transform: transforms = None,
        shuffle: bool = True,
        seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    frame_paths = frames_path.iterdir()
    lab_data, labels = get_labeled_data(root_path, annot_path, labels_to_idx)
    unlab_data = get_unlabeled_data(frames_path, lab_data)

    train_lab_data, train_unlab_data, test_data, \
    valid_data, train_labels, test_labels = split_data(
        lab_data = lab_data,
        unlab_data = unlab_data,
        labels = labels,
        train_lab_size = train_lab_size,
        test_size = test_size,
        train_unlab_size = train_unlab_size,
        val_size = val_size,
        shuffle = shuffle,
        seed = seed
    )
    train_lab_dataset = DreyeveDataset(
        data = train_lab_data,
        labels = train_labels,
        transform = transform,
        labels_to_idx = labels_to_idx
    )
    test_dataset = DreyeveDataset(
        data = test_data,
        labels = test_labels,
        transform = transform,
        labels_to_idx = labels_to_idx
    )
    train_unlab_dataset = DreyeveDataset(
        data = train_unlab_data,
        labels = None,
        transform = transform,
        labels_to_idx = labels_to_idx
    )
    valid_dataset = DreyeveDataset(
        data = valid_data,
        labels = None,
        transform = transform,
        labels_to_idx = labels_to_idx
    )
    return train_lab_dataset, test_dataset, train_unlab_dataset, valid_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type = str, required = True, help = "Path to the dataset: /..../Dr(eye)ve")
    parser.add_argument("--train_lab_size", type = float, default = 0.1, help = "Size of the labeled training set")
    parser.add_argument("--test_size", type = float, default = 0.9, help = "Size of the testing set")
    parser.add_argument("--train_unlab_size", type = float, default = 0.8, help = "Size of the unlabeled training set")
    parser.add_argument("--val_size", type = float, default = 0.2, help = "Size of the validation set")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    args = parser.parse_args()

    labels_to_idx = {
        "Dangerous": 0,
        "NOT Dangerous": 1
    }
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((108, 192), antialias = True)
    ])
    train_lab_dataset, test_dataset, train_unlab_dataset, valid_dataset = make_datasets(
        root_path = Path(args.root_dir),
        frames_path = Path(args.root_dir) / "data_frames",
        annot_path = Path(args.root_dir) / "data_annotations.json",
        train_lab_size = args.train_lab_size,
        test_size = args.test_size,
        train_unlab_size = args.train_unlab_size,
        val_size = args.val_size,
        transform = transform,
        labels_to_idx = labels_to_idx,
        shuffle = True,
        seed = args.seed
    )
    test_dataset.plot_sample(index = 1)
    plt.show()