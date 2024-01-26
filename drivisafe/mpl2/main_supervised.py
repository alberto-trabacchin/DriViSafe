from data import DATASET_GETTERS
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.amp.autocast_mode import autocast

import argparse
import logging
from vit_pytorch import SimpleViT
from tqdm import tqdm
import torch
import numpy as np
import random
import wandb
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='SimpleViT', type=str, help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--dataset', type=str, default='dreyeve', help='dataset name')
parser.add_argument('--model', type=str, default='simplevit', help='model name')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--total_steps", type=int, default=10000, help="number of training steps")
parser.add_argument("--eval_step", type=int, default=200, help="number of eval steps")
parser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="device (cuda or cpu)")
parser.add_argument("--amp", type=bool, default=True, help="use mixed precision training")
parser.add_argument("--subset_size", type=int, default=None, help="subset size")
parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
parser.add_argument("--num_train_lb", type=int, default=2)
parser.add_argument("--num_val", type=int, default=2)
parser.add_argument("--num_test", type=int, default=-1) # <-- Old: now not choosing test size. Just taking all remaining labeled samples.
parser.add_argument("--seed", type=int, default=42, help="random seed")

args = parser.parse_args()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate(model, test_loader, criterion, args):
    accuracies = []
    losses = []
    test_iter = tqdm(test_loader)
    with torch.inference_mode():
        for step, (images, targets) in enumerate(test_iter):
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with autocast(enabled = args.amp, device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            acc1 = (logits.argmax(dim=-1) == targets).float().sum() / batch_size
            accuracies.append(acc1.item())
            losses.append(loss.item())
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}."
            )
    
    mean_acc = statistics.mean(accuracies)
    mean_loss = statistics.mean(losses)
    test_iter.close()
    
    return mean_loss, mean_acc


def train_loop(model, lb_loader, val_loader, test_loader, finetune_loader, optimizer, criterion, args):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_train_lb}")
    logger.info(f"   Total steps = {args.total_steps}")
    lb_iter = iter(lb_loader)
    train_losses = []
    train_accuracies = []
    top1_acc = 0.0
    for step in range(args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
        model.train()
        try:
            images, targets = next(lb_iter)
        except:
            lb_iter = iter(lb_loader)
            images, targets = next(lb_iter)
        
        images = images.to(args.device)
        targets = targets.to(args.device)
        logits = model(images)
        loss = criterion(logits, targets)
        train_losses.append(loss.item())
        train_acc = (logits.argmax(dim=-1) == targets).float().sum() / args.batch_size
        train_accuracies.append(train_acc.item())
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}."
            f" train/loss: {loss.item():.6f}."
        )
        pbar.update()

        if (step + 1) % args.eval_step == 0:
            train_loss = train_losses[0]
            train_acc = train_accuracies[0]
            pbar.close()
            test_loss, test_acc = evaluate(model, test_loader, criterion, args)
            if test_acc > top1_acc:
                top1_acc = test_acc
                # save checkpoint
            print(f"test/loss: {test_loss:.4f}, test/acc: {test_acc:.4f}.")
            wandb.log({"train/loss": train_loss, 
                       "train/top1-acc": train_acc,
                       "test/loss": test_loss,
                       "test/acc": test_acc})
            train_losses = []
            train_accuracies = []


if __name__ == "__main__":
    set_seeds(args.seed)
    lb_dataset, _, val_dataset, test_dataset, finetune_dataset = DATASET_GETTERS[args.dataset](args)
    lb_loader = DataLoader(
        lb_dataset,
        sampler=RandomSampler(lb_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=RandomSampler(val_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=RandomSampler(test_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers
    )
    finetune_loader = DataLoader(
        finetune_dataset,
        sampler=RandomSampler(finetune_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers
    )

    if args.dataset == "dreyeve":
        if args.model == "simplevit":
            model = SimpleViT(
                image_size = (108, 192),
                patch_size = 6,
                num_classes = 2,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048
            )
    model = model.to(args.device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()
    wandb.init(name=f"{args.name}-{len(lb_dataset)}LB-{len(test_dataset)}TS", project='MPL-DriViSafe', config=args)
    train_loop(model, lb_loader, val_loader, test_loader, finetune_loader, optimizer, criterion, args)