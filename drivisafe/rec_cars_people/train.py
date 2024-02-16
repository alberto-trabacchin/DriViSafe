from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


if __name__ == "__main__":
    print(torch.cuda.is_available())