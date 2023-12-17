import argparse
import torch

from drivisafe import data_setup, utils, engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type = str, required = True, help = "Path to the dataset: /..../Dr(eye)ve")
    parser.add_argument("--train_lab_size", type = float, default = 0.1, help = "Size of the labeled training set")
    parser.add_argument("--test_size", type = float, default = 0.9, help = "Size of the testing set")
    parser.add_argument("--train_unlab_size", type = float, default = 0.8, help = "Size of the unlabeled training set")
    parser.add_argument("--val_size", type = float, default = 0.2, help = "Size of the validation set")
    parser.add_argument("--lr", type = float, default = 0.0001, help = "Learning rate for optimizer")
    parser.add_argument("--epochs", type = int, default = 100, help = "Number of epochs to train for")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    parser.add_argument("--shuffle", type = bool, default = True, help = "Whether to shuffle the data during training")
    parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers for the dataloader")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--device", type = str, default = "cuda" if torch.cuda.is_available() else "cpu", help = "Device to train on")
    args = parser.parse_args()

    # Set the random seed
    torch.manual_seed(args.seed)
    args.device = torch.device(args.device)
    print(args.device)
