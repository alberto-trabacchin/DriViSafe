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

parser = argparse.ArgumentParser()
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

args = parser.parse_args()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate(model, test_loader, criterion, args):
    accuracies = []
    test_iter = tqdm(test_loader)
    with torch.inference_mode():
        for step, (images, targets) in enumerate(test_iter):
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with autocast(enabled = args.amp):
                logits = model(images)
                loss = criterion(logits, targets)
            acc1 = (logits.argmax(dim=-1) == targets).float().sum() / batch_size
            accuracies.append(acc1)
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}."
                f"Test Acc: {acc1:.4f}. Test Loss: {loss.item():.4f}."
            )
    mean_acc = torch.mean(torch.stack(accuracies)).item()
    print(f"Mean Acc: {mean_acc:.4f}")
    test_iter.close()
    return loss.item(), mean_acc

def train_loop(model, lb_loader, val_loader, test_loader, finetune_loader, optimizer, criterion, args):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_train_lb}")
    logger.info(f"   Total steps = {args.total_steps}")
    for step in range(args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
        model.train()
        try:
            images, targets = next(lb_loader)
        except:
            print("Reloading lb_loader")
            lb_loader = iter(lb_loader)
            images, targets = next(lb_loader)
        
        images = images.to(args.device)
        targets = targets.to(args.device)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}."
            f" Train Loss: {loss.item():.4f}."
        )
        pbar.update()

        if (step + 1) % args.eval_step == 0:
            pbar.close()
            test_loss, top1 = evaluate(model, test_loader, criterion, args)


if __name__ == "__main__":
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

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()
    train_loop(model, lb_loader, val_loader, test_loader, finetune_loader, optimizer, criterion, args)