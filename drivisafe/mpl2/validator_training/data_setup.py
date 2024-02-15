from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.transforms import transforms
from PIL import Image
from typing import List, Dict
import json
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt


class DreyeveDataset(Dataset):
    """
    A custom Dataset class for the Dreyeve dataset.

    This class inherits from the PyTorch Dataset class and overrides the __init__ and __getitem__ methods.
    It represents a dataset of images with optional labels and transformations.

    Attributes:
        data (List[Path]): A list of paths to the images in the dataset.
        labels (List[int], optional): A list of labels corresponding to the images. Defaults to None.
        transform (transforms, optional): A torchvision.transforms instance containing the transformations to be applied to the images. Defaults to None.
        labels_to_idx (Dict[str, int], optional): A dictionary mapping labels to indices. Defaults to None.
    """
    def __init__(
            self,
            data: List[Path],
            labels: List[int] = None,
            transform: transforms = None,
            labels_to_idx: Dict[str, int] = None
    ) -> None:
        # self.data = data
        self.data = []
        for path in data:
            assert(path.exists())
            self.data.append(Image.open(path))
        self.data = np.stack(self.data, axis = 0)
        self.targets = labels
        self.transform = transform
        self.labels_to_idx = labels_to_idx
    
    def __getitem__(self, index: int) -> Tuple[str, int]:
        """
        Get the item at the given index.

        If transform is not None, apply it to the image.
        If labels is not None, return the image and its corresponding label.
        Otherwise, return the image and None.

        Args:
            index (int): The index of the item.

        Returns:
            Tuple[str, int]: The image and its label (or None).
        """
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.targets is not None:
            target = self.targets[index]
            target = torch.LongTensor([target]).squeeze()
            return image, target
        else:
            
            # Change return of label
            return image, torch.LongTensor([2]).squeeze()

    def __len__(self) -> int:
        return len(self.data)


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
        num_lb_train: int,
        num_test: int,
        num_val: int,
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
    assert(num_lb_train > 0)
    assert(num_test > 0)
    assert(num_val > 0)
    assert(len(lab_data) >= num_lb_train + num_test)
    assert(len(unlab_data) > num_val)

    # Shuffle the data
    if shuffle:
        lab_zip = list(zip(lab_data, labels))
        random.seed(seed)
        random.shuffle(lab_zip)
        random.shuffle(unlab_data)
        lab_data, labels = zip(*lab_zip)

    # Calculate the sizes of the sets
    train_lab_len = max(1, num_lb_train)
    test_len = max(1, num_test)
    val_len = max(1, num_val)
    train_unlab_len = len(unlab_data) - val_len
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
        train_lab_size: int,
        test_size: int,
        val_size: int,
        labels_to_idx: Dict[str, int],
        transform: transforms = None,
        shuffle: bool = True,
        seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create datasets for training (labeled + unlabeled), testing (labeled), and validation(unlabeled).

    This function first retrieves the labeled and unlabeled data, then splits them into labeled training,
    unlabeled training, testing, and validation datasets.
    It returns these datasets in a tuple.

    Args:
        root_path (Path): The root path.
        frames_path (Path): The path to the frames.
        annot_path (Path): The path to the annotations.
        train_lab_size (float): The size of the labeled training dataset.
        test_size (float): The size of the testing dataset.
        train_unlab_size (float): The size of the unlabeled training dataset.
        val_size (float): The size of the validation dataset.
        labels_to_idx (Dict[str, int]): The mapping from labels to indices.
        transform (transforms, optional): A torchvision.transforms instance containing the 
                                          transformations to be applied to the images. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The training, testing, and validation datasets.
    """
    lab_data, labels = get_labeled_data(root_path, annot_path, labels_to_idx)
    unlab_data = get_unlabeled_data(frames_path, lab_data)

    train_lab_data, train_unlab_data, test_data, \
    valid_data, train_labels, test_labels = split_data(
        lab_data = lab_data,
        unlab_data = unlab_data,
        labels = labels,
        num_lb_train = train_lab_size,
        num_test = test_size,
        num_val = val_size,
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
    return train_lab_dataset, train_unlab_dataset, test_dataset, valid_dataset


def make_dataloaders(
        lab_train_dataset: Dataset,
        unlab_train_dataset: Dataset,
        test_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int
) -> DataLoader:
    """
    Create dataloaders for training (labeled + unlabeled), testing (labeled), and validation(unlabeled).

    Args:
        lab_train_dataset (Dataset): The labeled training dataset.
        unlab_train_dataset (Dataset): The unlabeled training dataset.
        test_dataset (Dataset): The testing dataset.
        val_dataset (Dataset): The validation dataset.
        batch_size (int): The batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): The number of workers for the dataloader.

    Returns:
        DataLoader: The training, testing, and validation dataloaders.
    """
    lab_train_dl = DataLoader(
        dataset = lab_train_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers
    )
    unlab_train_dl = DataLoader(
        dataset = unlab_train_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers
    )
    test_dl = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers
    )
    val_dl = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers
    )
    return lab_train_dl, unlab_train_dl, test_dl, val_dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type = str, required = True, help = "Path to the dataset: /..../Dr(eye)ve")
    parser.add_argument("--num_lb_train", type = int, required=True, help = "Size of the labeled training set")
    parser.add_argument("--num_val", type = int, required=True, help = "Size of the validation set")
    parser.add_argument("--num_test", type = int, required=True, help = "Size of the testing set")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    parser.add_argument("--shuffle", type = bool, default = True, help = "Whether to shuffle the data during training")
    parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers for the dataloader")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    args = parser.parse_args()

    # Dict to map labels to indices
    labels_to_idx = {
        "Safe": 0,
        "Dangerous": 1
    }
    
    # Transform to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create the datasets
    train_lab_dataset, train_unlab_dataset, test_dataset, valid_dataset = make_datasets(
        root_path = Path(args.dataset_path),
        frames_path = Path(args.dataset_path) / "dreyeve",
        annot_path = Path(args.dataset_path) / "data_annotations.json",
        train_lab_size = args.num_lb_train,
        test_size = args.num_test,
        val_size = args.num_val,
        transform = transform,
        labels_to_idx = labels_to_idx,
        shuffle = True,
        seed = args.seed
    )

    # Create the dataloaders
    train_lab_dl, train_unlab_dl, test_dl, val_dl = make_dataloaders(
        lab_train_dataset = train_lab_dataset,
        unlab_train_dataset = train_unlab_dataset,
        test_dataset = test_dataset,
        val_dataset = valid_dataset,
        batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.num_workers
    )

    # Plot a sample image from the test dataset
    X, y = test_dataset[0]
    print("Sample shapes: ", X.shape, y.shape)
    print("Sample output", y)
    print("Sample data types:", type(X), type(y))

    images, targets = next(iter(test_dl))
    print("Batch targets: ", targets)

    print("\nTrain labeled dataset samples: ", len(train_lab_dataset))
    print("Train unlabeled dataset samples: ", len(train_unlab_dataset))
    print("Validation dataset samples: ", len(valid_dataset))
    print("Test dataset samples: ", len(test_dataset))