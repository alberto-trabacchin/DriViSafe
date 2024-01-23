import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing import List, Tuple
from torch.utils.data import Dataset
from pathlib import Path
from data_setup import DreyeveDataset
import json
from matplotlib import pyplot as plt

from augmentation import RandAugmentCIFAR
import argparse
import random

logger = logging.getLogger(__name__)

cifar10_mean = (0.491400, 0.482158, 0.4465231)
cifar10_std = (0.247032, 0.243485, 0.2615877)
cifar100_mean = (0.507075, 0.486549, 0.440918)
cifar100_std = (0.267334, 0.256438, 0.276151)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentCIFAR(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )
    finetune_dataset = CIFAR10SSL(
        args.data_path, finetune_idxs, train=True,
        transform=transform_finetune
    )
    train_unlabeled_dataset = CIFAR10SSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=TransformMPL(args, mean=cifar10_mean, std=cifar10_std)
    )

    test_dataset = datasets.CIFAR10(args.data_path, train=False,
                                    transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def get_cifar100(args):
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 10  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugmentCIFAR(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, finetune_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )
    finetune_dataset = CIFAR100SSL(
        args.data_path, finetune_idxs, train=True,
        transform=transform_fintune
    )
    train_unlabeled_dataset = CIFAR100SSL(
        args.data_path, train_unlabeled_idxs, train=True,
        transform=TransformMPL(args, mean=cifar100_mean, std=cifar100_std)
    )

    test_dataset = datasets.CIFAR100(args.data_path, train=False,
                                     transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def get_dreyeve(args):
    transform_labeled = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_finetune = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.subset_size is not None:
        subset_size = range(args.subset_size)
    else:
        subset_size = None
    base_dataset = DreyeveSSL(args.data_path, indexs = subset_size)

    train_lb_idxs, train_ul_idxs, val_idxs, \
    test_idxs, finetune_idxs = x_u_split_dreyeve(args, base_dataset.targets)


    train_labeled_dataset = DreyeveSSL(root = args.data_path,
                                       indexs = train_lb_idxs,
                                       transform = transform_labeled)
    train_unlabeled_dataset = DreyeveSSL(root = args.data_path,
                                         indexs = train_ul_idxs,
                                         transform = TransformDreyeveMPL(args, mean=normal_mean, std=normal_std))
    val_dataset = DreyeveSSL(root = args.data_path,
                             indexs = val_idxs,
                             transform = transform_val)
    test_dataset = DreyeveSSL(root = args.data_path,
                              indexs = test_idxs,
                              transform = transform_val)
    finetune_dataset = DreyeveSSL(root = args.data_path,
                                  indexs = finetune_idxs,
                                  transform = transform_finetune)

    # print("train_lb_idxs:\t", train_lb_idxs[0], "\t", train_lb_idxs.shape, "\t\t", base_dataset.full_img_names[train_lb_idxs[0]])
    # print("train_ul_idxs:\t", train_ul_idxs[0], "\t", train_ul_idxs.shape, "\t", base_dataset.full_img_names[train_ul_idxs[0]])
    # print("val_idxs:\t", val_idxs[0], "\t", val_idxs.shape, "\t\t", base_dataset.full_img_names[val_idxs[0]])
    # print("test_idxs:\t", test_idxs[0], "\t", test_idxs.shape, "\t\t", base_dataset.full_img_names[test_idxs[0]])
    print("train_lb len:\t", len(train_lb_idxs))
    print("train_ul len:\t", len(train_ul_idxs))
    print("val len:\t", len(val_idxs))
    print("test len:\t", len(test_idxs))

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset
    


def x_u_split_dreyeve(args, targets):
    train_lb, train_ul, val, test = [], [], [], []
    n_classes = args.num_classes
    train_lb_size = args.num_train_lb
    val_size = args.num_val
    test_size = args.num_test

    targets = np.array(targets)
    lb_idxs = np.where(targets != -1)[0]
    ul_idxs = np.where(targets == -1)[0]
    lb_size_sec = train_lb_size // n_classes
    test_size_sec = test_size // n_classes + lb_size_sec

    val, train_ul = np.split(ul_idxs, [val_size])
    train_ul = list(train_ul)
    for i in range(n_classes):
        lb_idxs = np.where(targets == i)[0]
        # train_lb_class, test_class, _ = np.split(lb_idxs, [lb_size_sec, test_size_sec]) # <-- Old split (test does not take all remaining data)
        train_lb_class, test_class = np.split(lb_idxs, [lb_size_sec]) # <-- New split (test takes all remaining data)
        train_lb.extend(train_lb_class)
        test.extend(test_class)
    train_ul.extend(train_lb)
    train_ul = np.array(train_ul)
    train_lb = np.array(train_lb)
    test = np.array(test)

    return train_lb, train_ul, val, test, train_lb
    

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    # iterate through all classes
    for i in range(args.num_classes):
        # get np array of indices of class i
        idx = np.where(labels == i)[0]
        # take label_per_class random indices from class i
        idx = np.random.choice(idx, label_per_class, False)
        # extend the list of indices of labeled data
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx_ex = np.hstack([labeled_idx for _ in range(num_expand_x)])
        np.random.shuffle(labeled_idx_ex)
        np.random.shuffle(labeled_idx)
        return labeled_idx_ex, unlabeled_idx, labeled_idx
    else:
        np.random.shuffle(labeled_idx)
        return labeled_idx, unlabeled_idx, labeled_idx


def x_u_split_test(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


class TransformDreyeveMPL(object):
    def __init__(self, args, mean, std):
        
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])
        
        self.base = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, x):
        aug = self.aug(x)
        return self.base(x), self.base(aug)


class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant'),
            RandAugmentCIFAR(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

class DreyeveSSL(Dataset):
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
    # First annotation: (27489, 01_4720.jpg)
    def __init__(
            self,
            root: str,
            transform: transforms = None,
            target_transform: transforms = None,
            indexs: List[int] = None,
            mode: str = None,
            args = None
    ) -> None:
        root = Path(root)
        self.imgs_path = root / "dreyeve"
        self.annot_path = root / "data_annotations.json"
        self.root = root
        self.full_img_names = [f.absolute() for f in self.imgs_path.iterdir() if f.is_file()]
        img_names = [f.name for f in self.full_img_names]
        self.targets = [-1] * len(img_names)

        annots = json.load(self.annot_path.open(mode = "r"))
        self.labels_to_idx = {"Safe": 0, "Dangerous": 1}
        self.idx_to_labels = {v: k for k, v in self.labels_to_idx.items()}
        # annot_targets = [self.labels_to_idx[d["choice"]] for d in annots] <-- To fix on labelstudio (empty labels)
        annot_targets = [self.labels_to_idx[d["choice"]] for d in annots if "choice" in d]
        # target_fnames = [d["image"][0].split("/")[-1] for d in annots] <-- To fix on labelstudio (empty labels)
        target_fnames = [d["image"][0].split("/")[-1] for d in annots if "choice" in d]
        for annot_id, annot_fname in enumerate(target_fnames):
            i = img_names.index(annot_fname)
            target = annot_targets[annot_id]
            self.targets[i] = target

        if indexs is not None:
            self.full_img_names = [self.full_img_names[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]

        self.data = []
        for path in self.full_img_names:
            self.data.append(Image.open(path))
        self.data = np.stack(self.data, axis = 0)
        self.transform = transform
        self.target_transform = target_transform
    

    def __getitem__(self, index: int) -> Tuple[str, int]:
        image = self.data[index]
        image = Image.fromarray(image)
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target


    def __len__(self) -> int:
        return len(self.data)
    
    def plot(self, index: int) -> None:
        image = self.data[index]
        target = self.targets[index]
        fig, ax = plt.subplots()
        if target in self.idx_to_labels:
            ax.set_title(self.idx_to_labels[target])
        else:
            ax.set_title("Unknown")
        ax.imshow(image)




DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'dreyeve': get_dreyeve}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MPL Implementation')
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_train_lb", type=int, default=2)
    parser.add_argument("--num_val", type=int, default=2)
    parser.add_argument("--num_test", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset_size", type=int, default=None)

    args = parser.parse_args()

    get_dreyeve(args)

