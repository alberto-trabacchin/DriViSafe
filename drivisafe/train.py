import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import random
import numpy as np
from pathlib import Path
from vit_pytorch import SimpleViT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.vit_for_small_dataset import ViT
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda import amp
import torch.nn as nn
import math
from tqdm import tqdm
import logging
import time
from torch.nn import functional as F
from drivisafe.utils import AverageMeter
from matplotlib import pyplot as plt
from typing import Tuple

from drivisafe import data_setup, utils

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type = str, required = True, help = "Path to the dataset: /..../Dr(eye)ve")
parser.add_argument("--train_lab_size", default = 0.5, type = float, help = "number of labeled train samples")
parser.add_argument("--test_size", default = 0.5, type = float, help = "number of test samples")
parser.add_argument("--train_unlab_size", default = 0.99, type = float, help = "number of unlabeled train samples")
parser.add_argument("--val_size", default = 0.01, type = float, help = "number of validation samples")
parser.add_argument("--total_steps", default = 1000, type = int, help = "number of total steps to run")
parser.add_argument("--eval_step", default = 100, type = int, help = "number of eval steps to run")
parser.add_argument("--warmup_steps", default = 0, type = int, help = "number of warmup steps to run")
parser.add_argument("--workers", default = 4, type = int, help = "number of workers")
parser.add_argument("--num_classes", default = 2, type = int, help = "number of classes")
parser.add_argument("--resize", default = 32, type = int, help = "resize image")
parser.add_argument("--batch-size", default = 16, type = int, help = "train batch size")
parser.add_argument("--teacher-dropout", default = 0, type = float, help = "dropout on last dense layer")
parser.add_argument("--student-dropout", default = 0, type = float, help = "dropout on last dense layer")
parser.add_argument("--teacher_lr", default = 0.01, type = float, help = "train learning rate")
parser.add_argument("--student_lr", default = 0.01, type = float, help = "train learning rate")
parser.add_argument("--device", default = "cuda" if torch.cuda.is_available() else "cpu", type = str, help = "cuda, cpu")
parser.add_argument("--momentum", default = 0.9, type = float, help = "SGD Momentum")
parser.add_argument("--nesterov", action = "store_true", help = "use nesterov")
parser.add_argument("--weight-decay", default = 0, type = float, help = "train weight decay")
parser.add_argument("--ema", default = 0, type = float, help = "EMA decay rate")
parser.add_argument("--warmup-steps", default = 0, type = int, help = "warmup steps")
parser.add_argument("--student-wait-steps", default = 0, type = int, help = "warmup steps")
parser.add_argument("--grad-clip", default = 1e9, type = float, help = "gradient norm clipping")
parser.add_argument("--shuffle", default = True, help = "shuffle train data")
parser.add_argument("--seed", default = 42, type = int, help = "seed for initializing training")
parser.add_argument("--label-smoothing", default = 0, type = float, help = "label smoothing alpha")
parser.add_argument("--mu", default = 7, type = int, help = "coefficient of unlabeled batch size")
parser.add_argument("--threshold", default = 0.95, type = float, help = "pseudo label threshold")
parser.add_argument("--temperature", default = 1, type = float, help = "pseudo label temperature")
parser.add_argument("--lambda-u", default = 1, type = float, help = "coefficient of unlabeled loss")
parser.add_argument("--uda-steps", default = 1, type = float, help = "warmup steps of lambda-u")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_loop(args, labeled_loader, unlabeled_loader, test_loader, finetune_dataset,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    logger.info("***** Running Training *****")
    logger.info(f"   Task =moon")
    logger.info(f"   Total steps = {args.total_steps}")

    # if args.world_size > 1:
    #     labeled_epoch = 0
    #     unlabeled_epoch = 0
    #     labeled_loader.sampler.set_epoch(labeled_epoch)
    #     unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    # for author's code formula
    # moving_dot_product = torch.empty(1).to(args.device)
    # limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    # nn.init.uniform_(moving_dot_product, -limit, limit)

    # for step in range(args.total_steps):
    for step in tqdm(range(args.total_steps)):
        if step % args.eval_step == 0:
            # pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            # pbar = tqdm(range(args.eval_step), disable=False)
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            # error occurs ↓
            # images_l, targets = labeled_iter.next()
            images_l, targets = next(labeled_iter)
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            # error occurs ↓
            # images_l, targets = labeled_iter.next()
            images_l, targets = next(labeled_iter)

        try:
            # error occurs ↓
            # (images_uw, images_us), _ = unlabeled_iter.next()
            images_u, _ = next(unlabeled_iter)
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            # error occurs ↓
            # (images_uw, images_us), _ = unlabeled_iter.next()
            images_u, _ = next(unlabeled_iter)

        data_time.update(time.time() - end)

        images_l = images_l.to(args.device)
        images_u = images_u.to(args.device)
        targets = targets.to(args.device)
        # print("targets: ", targets)
        with amp.autocast(enabled=args.amp):
            batch_size = images_l.shape[0]
            t_images = torch.cat((images_l, images_u))
            t_logits = teacher_model(t_images)
            t_logits_l = t_logits[:batch_size]
            t_logits_u = t_logits[batch_size:]
            del t_logits

            t_loss_l = criterion(t_logits_l, targets)
            # print("t_loss_l: ", t_loss_l.item())

            soft_pseudo_label = torch.softmax(t_logits_u.detach() / args.temperature, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            # print(max_probs, hard_pseudo_label, soft_pseudo_label)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_u, dim=-1)).sum(dim=-1) * mask
            )
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_images = torch.cat((images_l, images_u))
            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, hard_pseudo_label)
            # print("s_loss: ", s_loss.item())

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        if args.ema > 0:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(images_l)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)

            # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
            # dot_product = s_loss_l_old - s_loss_l_new

            # author's code formula
            dot_product = s_loss_l_new - s_loss_l_old
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product

            _, hard_pseudo_label = torch.max(t_logits_u.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_u, hard_pseudo_label)
            # test
            # t_loss_mpl = torch.tensor(0.).to(args.device)
            t_loss = t_loss_uda + t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        # if args.world_size > 1:
        #     s_loss = reduce_tensor(s_loss.detach(), args.world_size)
        #     t_loss = reduce_tensor(t_loss.detach(), args.world_size)
        #     t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
        #     t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
        #     t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
        #     mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        test_loss, test_acc = evaluate(
            model = student_model,
            data_loader = test_loader,
            criterion = torch.nn.CrossEntropyLoss(),
            device = args.device
        )

        tqdm.write(
            f"Teacher Train Loss: {t_losses.avg:.10f}\n" \
            f"Student Train Loss: {s_losses.avg:.10f}\n" \
            f"Teacher Train Loss MPL: {t_losses_mpl.avg:.10f}\n" \
            f"Student Test Loss: {test_loss.avg:.10f}\n" \
            f"Student Test Accuracy: {test_acc * 100:.4f}%\n"
        )

        batch_time.update(time.time() - end)
        # pbar.set_description(
        #     f"Train Iter: {step+1:3}/{args.total_steps:3}. "
        #     f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
        #     f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
        #     f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        # pbar.update()
        # if args.local_rank in [-1, 0]:
        #     args.writer.add_scalar("lr", get_lr(s_optimizer), step)
            # wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step // args.eval_step
        # if (step + 1) % args.eval_step == 0:
            # pbar.close()
            # if args.local_rank in [-1, 0]:
            #     args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
            #     args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
            #     args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
            #     args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
            #     args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
            #     args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                # wandb.log({"train/1.s_loss": s_losses.avg,
                #            "train/2.t_loss": t_losses.avg,
                #            "train/3.t_labeled": t_losses_l.avg,
                #            "train/4.t_unlabeled": t_losses_u.avg,
                #            "train/5.t_mpl": t_losses_mpl.avg,
                #            "train/6.mask": mean_mask.avg})

                # test_model = avg_student_model if avg_student_model is not None else student_model
                # test_loss, top1 = evaluate(args, test_loader, test_model, criterion)

                # args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                # args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                # wandb.log({"test/loss": test_loss,
                #            "test/acc@1": top1})

                # is_best = top1 > args.best_top1
                # if is_best:
                #     args.best_top1 = top1

                # logger.info(f"top-1 acc: {top1:.2f}")
                # logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                # save_checkpoint(args, {
                #     'step': step + 1,
                #     'teacher_state_dict': teacher_model.state_dict(),
                #     'student_state_dict': student_model.state_dict(),
                #     'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                #     'best_top1': args.best_top1,
                #     'teacher_optimizer': t_optimizer.state_dict(),
                #     'student_optimizer': s_optimizer.state_dict(),
                #     'teacher_scheduler': t_scheduler.state_dict(),
                #     'student_scheduler': s_scheduler.state_dict(),
                #     'teacher_scaler': t_scaler.state_dict(),
                #     'student_scaler': s_scaler.state_dict(),
                # }, is_best)

    # if args.local_rank in [-1, 0]:
    #     args.writer.add_scalar("result/test_acc@1", args.best_top1)
    #     wandb.log({"result/test_acc@1": args.best_top1})


def validate(
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device
) -> None:
    predictions = []
    for batch in data_loader:
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim = 1)
        preds = torch.argmax(probs, dim = 1)
        predictions.extend(preds.to("cpu").numpy())
    return predictions


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> float:
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.inference_mode():
        for batch in data_loader:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim = 1)
            preds = torch.argmax(probs, dim = 1)
            total_accuracy += torch.sum(preds == targets).item()
    total_loss = total_loss / len(data_loader)
    total_accuracy = total_accuracy / len(data_loader.dataset)
    return total_loss, total_accuracy


if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed)
    args.device = torch.device(args.device)

    labels_to_idx = {
        "Dangerous": 0,
        "NOT Dangerous": 1
    }
    idx_to_labels = {
        0: "Dangerous",
        1: "NOT Dangerous"
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size = (32, 32), antialias = True)
    ])

    train_lab_dataset, train_unlab_dataset, test_dataset, valid_dataset = data_setup.make_datasets(
        root_path = Path(args.dataset_path),
        frames_path = Path(args.dataset_path) / "data_frames",
        annot_path = Path(args.dataset_path) / "data_annotations.json",
        train_lab_size = args.train_lab_size,
        test_size = args.test_size,
        train_unlab_size = args.train_unlab_size,
        val_size = args.val_size,
        transform = transform,
        labels_to_idx = labels_to_idx,
        shuffle = True,
        seed = args.seed
    )

    train_lab_dl, train_unlab_dl, test_dl, val_dl = data_setup.make_dataloaders(
        lab_train_dataset = train_lab_dataset,
        unlab_train_dataset = train_unlab_dataset,
        test_dataset = test_dataset,
        val_dataset = valid_dataset,
        batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.workers
    )

    teacher_model = ViT(
        image_size = args.resize,
        patch_size = 32,
        num_classes = args.num_classes,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
    )
    student_model = ViT(
        image_size = args.resize,
        patch_size = 32,
        num_classes = args.num_classes,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
    )

    teacher_model.to(args.device)
    student_model.to(args.device)
    criterion = utils.create_loss_fn(args)

    no_decay = ['bn']
    teacher_parameters = [
        {'params': [p for n, p in teacher_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in teacher_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    student_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in student_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_optimizer = torch.optim.SGD(teacher_parameters,
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)
    s_optimizer = torch.optim.SGD(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)
    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)
    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(
        args = args,
        labeled_loader = train_lab_dl,
        unlabeled_loader = train_unlab_dl,
        test_loader = test_dl,
        finetune_dataset = None,
        teacher_model = teacher_model,
        student_model = student_model,
        avg_student_model = None,
        criterion = criterion,
        t_optimizer = t_optimizer,
        s_optimizer = s_optimizer,
        t_scheduler = t_scheduler,
        s_scheduler = s_scheduler,
        t_scaler = t_scaler,
        s_scaler = s_scaler
    )
    
    val_ouptut = validate(
        model = student_model,
        data_loader = val_dl,
        device = args.device
    )
    for i, out in enumerate(val_ouptut):
        valid_dataset.plot_sample(i, title = f"Prediction: {idx_to_labels[out]}")
        plt.show()

    test_loss, test_acc = evaluate(
        model = student_model,
        data_loader = test_dl,
        criterion = torch.nn.CrossEntropyLoss(),
        device = args.device
    )
    print(f"Test loss: {test_loss :.8f}")
    print(f"Test accuracy: {test_acc * 100 :.4f}")