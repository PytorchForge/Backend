# --- IMPORTS (imports.j2) ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms

# --- MODELS (models/layers.j2) ---


# --- TRANSFORMS (data/transforms/image.j2) ---


# --- LOADERS (data/loaders/image/classification.j2) ---
# PLACEHOLDER


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(class_dir)

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(sorted(set(self.labels)))
        }
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_image_classification_loaders(
    data_dir, batch_size, transform_train=None, transform_val=None
):
    train_dataset = CustomImageDataset(
        os.path.join(data_dir, "train"), transform=transform_train
    )
    val_dataset = CustomImageDataset(
        os.path.join(data_dir, "val"), transform=transform_val
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


import random

import numpy as np

# --- setup.j2 ---
import torch


def set_seed(seed=None):
    seed = seed or config.training.get("seed", {{config.get("SEED", 42)}})
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device=None):
    device_str = device or config.get("DEVICE", "cpu")
    return torch.device(device_str if torch.cuda.is_available() else "cpu")


# --- train/utils.j2 ---
import torch.nn as nn
import torch.optim as optim


def get_criterion(config):
    loss_cfg = config.loss
    name = loss_cfg.name
    params = loss_cfg.get("params", {})
    return getattr(nn, name)(**params)


def get_optimizer(model, config):
    opt = config.optimizer
    Optim = getattr(optim, opt.name)
    params = opt.get("params", {})
    return Optim(model.parameters(), lr=config.training.learningRate, **params)


# --- train/train_loop.j2 ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for batch in loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


# --- train/eval_loop.j2 ---
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)
