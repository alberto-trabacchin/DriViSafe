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
from torch.utils.data import DataLoader

from augmentation import RandAugmentCIFAR
import argparse
import random
import torch
import os
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights


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
        transform=transform_finetune
    )
    train_unlabeled_dataset = CIFAR100SSL(
        args.data_path, train_unlabeled_idxs, train=True,
        transform=TransformMPL(args, mean=cifar100_mean, std=cifar100_std)
    )

    test_dataset = datasets.CIFAR100(args.data_path, train=False,
                                     transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, finetune_dataset


def get_dreyeve(args):

    def dangerous_scene(prediction):
        targets = prediction["labels"]
        accs = prediction["scores"]
        lb2idx = {
            "person": 1,
            "car": 3,
            "motorcycle": 4,
            "bus": 6,
            "truck": 8
        }
        for t, acc in zip(targets, accs):
            if (t in lb2idx.values()) and acc > 0.5:
                return True
            
        return False


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
    base_dataset = DreyeveSSL(args, args.data_path, indexs = subset_size, targets_name=args.targets_name, mode="train")

    train_lb_idxs, train_ul_idxs, finetune_idxs = x_u_split_dreyeve(args, base_dataset.targets)

    train_labeled_dataset = DreyeveSSL(args,
                                       root = args.data_path,
                                       targets_name = args.targets_name,
                                       indexs = train_lb_idxs,
                                       transform = transform_labeled,
                                       mode = "train")
    train_unlabeled_dataset = DreyeveSSL(args,
                                         root = args.data_path,
                                         targets_name = args.targets_name,
                                         indexs = train_ul_idxs,
                                         transform = TransformDreyeveMPL(args, mean=normal_mean, std=normal_std),
                                         mode = "train")
    val_dataset = DreyeveSSL(args,
                             root = args.data_path,
                             targets_name = args.targets_name,
                             transform = transform_val,
                             mode = "val")
    test_dataset = DreyeveSSL(args,
                              root = args.data_path,
                              targets_name = args.targets_name,
                              transform = transform_val,
                              mode = "test")
    finetune_dataset = DreyeveSSL(args,
                                  root = args.data_path,
                                  targets_name = args.targets_name,
                                  indexs = finetune_idxs,
                                  transform = transform_finetune)
    


    # Add validator data
    if args.validator is not None:
        val_idxs = np.setdiff1d(train_ul_idxs, train_lb_idxs)
        val_idxs = np.random.choice(val_idxs, size = args.validator, replace = False)
        images = []
        for i in list(val_idxs):
            img, _ = base_dataset[i]
            img = transform_labeled(img)
            images.append(img)
        val_images = torch.stack(images)
        val_model = retinanet_resnet50_fpn_v2(weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        val_model = val_model.to(args.device)
        val_model.eval()

        val_targets = []
        predictions = []
        for batch_val_images in torch.split(val_images, args.batch_size):
            with torch.inference_mode():
                batch_val_images = batch_val_images.to(args.device)
                pred_val = val_model(batch_val_images)
                predictions.extend(pred_val)

        for p in predictions:
            if dangerous_scene(p):
                val_targets.append(base_dataset.labels_to_idx["Dangerous (TRAIN)"])
            else:
                val_targets.append(base_dataset.labels_to_idx["Safe (TRAIN)"])
        
        assert(val_idxs.size == len(val_targets))

        val_images = val_images.cpu().numpy()
        data = [train_labeled_dataset.data, val_images.transpose(0, 2, 3, 1)]
        train_labeled_dataset.data = np.vstack(data)
        train_labeled_dataset.targets.extend(val_targets)

    print("train_lb len:\t", len(train_labeled_dataset))
    print("train_ul len:\t", len(train_unlabeled_dataset))
    print("val len:\t", len(val_dataset))
    print("test len:\t", len(test_dataset))
    print("validator-safe:\t", sum(1 for n in val_targets if n==0))
    print("validator-dang:\t", sum(1 for n in val_targets if n==1))
    exit()
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, finetune_dataset


def x_u_split_dreyeve(args, targets):
    max_labels1 = sum(1 for n in targets if n==0)
    max_labels2 = sum(1 for n in targets if n==1)
    max_labels = 2 * min(max_labels1, max_labels2)

    if args.num_labeled is None:
        num_labeled = max_labels
    else:
        num_labeled = max(args.num_labeled, max_labels)
    label_per_class = num_labeled // args.num_classes
    labels = np.array(targets)
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
    assert (len(labeled_idx) == num_labeled, f"len(labeled_idx) = {len(labeled_idx)}, num_labeled = {num_labeled}")
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx, labeled_idx
    

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
            args,
            root: str,
            targets_name: str,
            transform: transforms = None,
            target_transform: transforms = None,
            indexs: List[int] = None,
            mode: str = "train"
    ) -> None:
        
        def is_train_label(target) -> bool:
            labels = list(self.labels_to_idx.keys())
            return (target == labels[0] or target == labels[1])
        
        def get_target_fname(annot) -> str:
            return annot["task"]["data"]["image"].split("/")[-1]
        
        root = Path(root)
        self.imgs_path = root / "dreyeve" / "images"
        self.targets_path = root / "dreyeve" / "targets" / targets_name
        self.root = root
        self.mode = mode
        self.all_data_paths = [f.absolute() for f in self.imgs_path.iterdir() if f.is_file()]
        self.all_data_fnames = [f.name for f in self.all_data_paths]
        self.targets = []

        self.labels_to_idx = {"Safe (TRAIN)": 0, "Dangerous (TRAIN)": 1, "Safe (TEST)": 0, "Dangerous (TEST)": 1, "Validation": 2}
        self.idx_to_labels = {0: "Safe", 1: "Dangerous"}

        json_files = [f for f in self.targets_path.iterdir() if f.is_file()]
        train_targets = []
        test_targets = []
        train_targets_fnames = []
        test_targets_fnames = []

        for f in json_files:
            annot = json.load(f.open(mode = "r"))
            if annot["result"]:
                target = annot["result"][0]["value"]["choices"][0]
                if (is_train_label(target)):
                    train_targets.append(self.labels_to_idx[target])
                    train_targets_fnames.append(get_target_fname(annot))
                elif (not is_train_label(target)):
                    test_targets.append(self.labels_to_idx[target])
                    test_targets_fnames.append(get_target_fname(annot))

        self.data_fnames = []
        self.data_paths = []
        self.targets = []
        if self.mode == "test":
            data_idxs = [self.all_data_fnames.index(tfn) for tfn in test_targets_fnames]
            self.data_fnames = [self.all_data_fnames[i] for i in data_idxs]
            self.data_paths = [self.all_data_paths[i] for i in data_idxs]
            self.targets = test_targets
        
        elif self.mode == "train":
            self.data_fnames = self.all_data_fnames.copy()
            self.data_paths = self.all_data_paths.copy()
            for tfn in test_targets_fnames:
                i = self.data_fnames.index(tfn)
                self.data_fnames.pop(i)
                self.data_paths.pop(i)
            self.targets = [-1] * len(self.data_fnames)

            for t, tfn in zip(train_targets, train_targets_fnames):
                i = self.data_fnames.index(tfn)
                self.targets[i] = t

            # if args.validator is not None:
            #     i = self.targets.index(-1)
            #     val_i = random.sample(i, counts = args.validator)
            #     val_data_paths = [self.all_data_paths[i] for i in val_i]
            #     val_model = retinanet_resnet50_fpn_v2(weitghts = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
            #     val_model = val_model.to(args.device)
            #     val_data = []
            #     for vdp in val_data_paths:
            #         val_data.append(Image.open(vdp))
                




        elif self.mode == "val":
            self.data_fnames = self.all_data_fnames.copy()
            self.data_paths = self.all_data_paths.copy()
            for tfn in test_targets_fnames:
                i = self.data_fnames.index(tfn)
                self.data_fnames.pop(i)
                self.data_paths.pop(i)
            for tfn in train_targets_fnames:
                i = self.data_fnames.index(tfn)
                self.data_fnames.pop(i)
                self.data_paths.pop(i)

            samples = random.sample(list(zip(self.data_fnames, self.data_paths)), args.num_val)
            self.data_fnames, self.data_paths = zip(*samples)
            self.data_fnames = list(self.data_fnames)
            self.data_paths = list(self.data_paths)
            self.targets = [-1] * len(self.data_fnames)
        

        if indexs is not None:
            self.data_fnames = [self.data_fnames[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]
            self.data_paths = [self.data_paths[i] for i in indexs]

        self.data = []
        for path in self.data_paths:
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
    parser.add_argument("--targets-name", type=str, default="cars-people")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_labeled", type=int, default=4, help="Multiple of 2")
    parser.add_argument("--num_val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset_size", type=int, default=None)

    args = parser.parse_args()

    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, finetune_dataset = get_dreyeve(args)
    train_lb_loader = DataLoader(
        dataset = train_labeled_dataset,
        batch_size = 2,
        shuffle = True,
        num_workers = 0
    )
    lb_iter = iter(train_lb_loader)
    images, targets = next(lb_iter)
    print(targets)