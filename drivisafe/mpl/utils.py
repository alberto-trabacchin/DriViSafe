import logging
import os
import shutil
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def create_loss_fn(args):
    # if args.label_smoothing > 0:
    #     criterion = SmoothCrossEntropyV2(alpha=args.label_smoothing)
    # else:  
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    return criterion.to(args.device)


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def model_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)


def save_checkpoint(args, state, is_best, finetune=False):
    os.makedirs(args.save_path, exist_ok=True)
    if finetune:
        name = f'{args.name}_finetune'
    else:
        name = args.name
    filename = f'{args.save_path}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{args.save_path}/{args.name}_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]
    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_moon(output, target):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    batch_size = target.shape[0]
    acc = torch.sum(torch.argmax(output, dim=1) == target).float() / batch_size
    return acc * 100


def plot_decision_boundary(model: torch.nn.Module, test_dataset, labeled_dataset):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    X_train, y_train = labeled_dataset.data, labeled_dataset.targets
    X_test, y_test = test_dataset.data, test_dataset.targets
    model.to("cpu")
    X_train, y_train = torch.FloatTensor(X_train).to("cpu"), torch.LongTensor(y_train).to("cpu")
    X_test, y_test = torch.FloatTensor(X_test).to("cpu"), torch.LongTensor(y_test).to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = torch.min(X_test[:, 0].min(), X_train[:, 0].min()), \
                             torch.max(X_test[:, 0].max(), X_train[:, 0].max())
    y_min, y_max = torch.min(X_test[:, 1].min(), X_train[:, 1].min()), \
                             torch.max(X_test[:, 1].max(), X_train[:, 1].max())
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, cmap=plt.cm.RdYlBu)
    plt.scatter(X_train[:, 0], X_train[:, 1], c="goldenrod", marker = "*", s=150)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_data(labeled_dataset, unlabeled_dataset, test_dataset):
    X_labeled, y_labeled = labeled_dataset.data, labeled_dataset.targets
    X_unlabeled, y_unlabeled = unlabeled_dataset.data, unlabeled_dataset.targets
    X_test, y_test = test_dataset.data, test_dataset.targets
    fig, axis = plt.subplots(1, 2, figsize=(12, 5))
    axis[0].scatter(X_unlabeled[:, 0], X_unlabeled[:, 1], c=y_unlabeled, cmap=plt.cm.RdYlBu, s=20)
    axis[0].scatter(X_labeled[:, 0], X_labeled[:, 1], c="goldenrod", marker = "*", s=150)
    axis[0].set_title("Training Data")
    axis[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, s=20)
    axis[1].scatter(X_labeled[:, 0], X_labeled[:, 1], c="goldenrod", marker = "*", s=150)
    axis[1].set_title("Test Data")


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        if self.alpha == 0:
            loss = F.cross_entropy(logits, labels)
        else:
            num_classes = logits.shape[-1]
            alpha_div_k = self.alpha / num_classes
            target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
                (1. - self.alpha) + alpha_div_k
            loss = (-(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)).mean()
        return loss


class SmoothCrossEntropyV2(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, label_smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        assert label_smoothing < 1.0
        self.smoothing = label_smoothing
        self.confidence = 1. - label_smoothing

    def forward(self, x, target):
        if self.smoothing == 0:
            loss = F.cross_entropy(x, target)
        else:
            logprobs = F.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count