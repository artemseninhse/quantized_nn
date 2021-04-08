import numpy as np
import random
import torch

from torchvision import datasets, transforms

SEED = 1234567890
CROP_SIZE = 28
NUM_WORKERS = 0

DATASETS = {
    "cifar10": datasets.CIFAR10,
    "fashionmnist": datasets.FashionMNIST,
    "mnist": datasets.MNIST,
    "voc-detect": datasets.VOCDetection,
    "voc-segm": datasets.VOCSegmentation
}

transforms_train = transforms.Compose([
              transforms.ToTensor(),
              transforms.RandomCrop(CROP_SIZE),
              # transforms.GaussianBlur(1),
              # transforms.RandomAdjustSharpness(0.9, p=0.5),
              transforms.RandomPerspective(p=0.5)
              ])

transforms_test = transforms.Compose([
              transforms.ToTensor()
              ])

TRANSFORMS = {
    "train": transforms_train,
    "test": transforms_test
}


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
