import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from typing import Tuple, Dict, List
from pathlib import Path
from tqdm.auto import tqdm
import os
import argparse

from drivisafe import data_setup, utils


def train_step(
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    model.train()
    train_loss = 0
    train_accuracy = 0
    for batch_sample in data_loader:
        X, y = batch_sample
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_accuracy += utils.accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    return train_loss, train_accuracy


def test_step(
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.inference_mode():
        for batch_sample in data_loader:
            X, y = batch_sample
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_accuracy += utils.accuracy(y_pred, y)
        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)
    return test_loss, test_accuracy


def train(
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device,
        verbose: bool = True
) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch": []
    }
    train_accuracy = 0
    test_accuracy = 0
    train_loss = 0
    test_loss = 0
    for epoch in tqdm(range(epochs)):
        model = model.to(device)
        train_loss, train_accuracy = train_step(
            model = model,
            data_loader = train_dataloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = device
        )
        test_loss, test_accuracy = test_step(
            model = model,
            data_loader = test_dataloader,
            loss_fn = loss_fn,
            device = device
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_accuracy)
        results["epoch"].append(epoch + 1)
        if verbose:
            results_table = utils.make_table_train_results(
                model_name = model.name,
                epoch = epoch + 1,
                train_loss = train_loss,
                test_loss = test_loss,
                train_acc = train_accuracy,
                test_acc = test_accuracy
            )
            tqdm.write(results_table.get_string())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 100, help = "Number of epochs to train for")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size to train with")
    parser.add_argument("--lr", type = float, default = 0.0001, help = "Learning rate for optimizer")
    parser.add_argument("--root_dir", type = str, required = True, help = "Path to the dataset: /..../Dr(eye)ve")
    parser.add_argument("--train_lab_size", type = float, default = 0.1, help = "Size of the labeled training set")
    parser.add_argument("--test_size", type = float, default = 0.9, help = "Size of the testing set")
    parser.add_argument("--train_unlab_size", type = float, default = 0.8, help = "Size of the unlabeled training set")
    parser.add_argument("--val_size", type = float, default = 0.2, help = "Size of the validation set")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size")
    parser.add_argument("--shuffle", type = bool, default = True, help = "Whether to shuffle the data during training")
    parser.add_argument("--num_workers", type = int, default = 4, help = "Number of workers for the dataloader")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    args = parser.parse_args()

    # Initialize constants
    DS_DOWNL_PATH = Path("data/")
    DS_PATH = DS_DOWNL_PATH / "pizza_steak_sushi"
    NUM_WORKERS = os.cpu_count()
    IN_SIZE = 3
    HID_SIZE = 10
    OUT_SIZE = 3
    EPOCHS = 100
    BATCH_SIZE = 32
    LR = 0.0001
    RESIZE = (512, 512)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloaders
    data_setup.download_dataset(DS_DOWNL_PATH, DS_PATH)
    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor()
    ])
    train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(
        train_dir = DS_PATH / "train",
        test_dir = DS_PATH / "test",
        transform = transform,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )

    # Create model, loss function and optimizer for training
    model = model_builder.TinyVGG(
        input_shape = IN_SIZE,
        hidden_units = HID_SIZE,
        output_shape = OUT_SIZE,
        name = "TinyVGG"
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LR)

    # Train model
    results = train(
        model = model,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = EPOCHS,
        device = device,
        verbose = True
    )
    utils.print_models_results(models_results = {model.name: results})