import argparse
import torch
import random
import numpy as np
from pathlib import Path

from drivisafe import data_setup, utils, engine


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type = str, required = True, help = "Path to the dataset: /..../Dr(eye)ve")
parser.add_argument("--epochs", default = 1000, type = int, help = "number of total steps to run")
parser.add_argument("--workers", default = 4, type = int, help = "number of workers")
parser.add_argument("--num-classes", default = 2, type = int, help = "number of classes")
parser.add_argument("--resize", default = 32, type = int, help = "resize image")
parser.add_argument("--batch-size", default = 64, type = int, help = "train batch size")
parser.add_argument("--teacher-dropout", default = 0, type = float, help = "dropout on last dense layer")
parser.add_argument("--student-dropout", default = 0, type = float, help = "dropout on last dense layer")
parser.add_argument("--teacher_lr", default = 0.01, type = float, help = "train learning rate")
parser.add_argument("--student_lr", default = 0.01, type = float, help = "train learning rate")
parser.add_argument("--momentum", default = 0.9, type = float, help = "SGD Momentum")
parser.add_argument("--nesterov", action = "store_true", help = "use nesterov")
parser.add_argument("--weight-decay", default = 0, type = float, help = "train weight decay")
parser.add_argument("--ema", default = 0, type = float, help = "EMA decay rate")
parser.add_argument("--warmup-steps", default = 0, type = int, help = "warmup steps")
parser.add_argument("--student-wait-steps", default = 0, type = int, help = "warmup steps")
parser.add_argument("--grad-clip", default = 1e9, type = float, help = "gradient norm clipping")
parser.add_argument("--seed", default = None, type = int, help = "seed for initializing training")
parser.add_argument("--label-smoothing", default = 0, type = float, help = "label smoothing alpha")
parser.add_argument("--mu", default = 7, type = int, help = "coefficient of unlabeled batch size")
parser.add_argument("--threshold", default = 0.95, type = float, help = "pseudo label threshold")
parser.add_argument("--temperature", default = 1, type = float, help = "pseudo label temperature")
parser.add_argument("--lambda-u", default = 1, type = float, help = "coefficient of unlabeled loss")
parser.add_argument("--uda-steps", default = 1, type = float, help = "warmup steps of lambda-u")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed)

    # Create the datasets
    train_lab_dataset, train_unlab_dataset, test_dataset, valid_dataset = make_datasets(
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
