import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        label_type="folder-name",
        label_map=None,
        return_format="dict",
        split_type="include",
    ):
        self.root = Path(root) / split
        self.label_map = self._infer_label_map()
        self.return_format = return_format
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.data = self._load_data()

    def _infer_label_map(self):
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        return {cls: i for i, cls in enumerate(classes)}

    def _load_data(self):
        items = []
        for cls in self.label_map:
            for img_path in (self.root / cls).glob("*.jpg"):
                items.append((img_path, self.label_map[cls]))
        return items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))

        return {"image": image, "label": label}


def get_loader(root, batch_size=32, split="train", shuffle=True, **kwargs):
    dataset = ImageClassificationDataset(root=root, split=split, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def set_seed(config, seed=None):
    seed = seed or config["training"].get("seed", config.get("SEED", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config, device=None):
    device_str = device or config.get("DEVICE", "cpu")
    return torch.device(device_str if torch.cuda.is_available() else "cpu")


def get_criterion(config):
    loss_cfg = config["training"]["loss"]
    name = loss_cfg["name"]
    params = loss_cfg.get("params", {})
    return getattr(nn, name)(**params)


def get_optimizer(model, config):
    opt_cfg = config["training"]["optimizer"]
    Optim = getattr(optim, opt_cfg["name"])
    params = opt_cfg.get("params", {})
    learning_rate = config["training"]["learning_rate"]
    return Optim(model.parameters(), lr=learning_rate, **params)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for batch in loader:
        inputs, targets = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == targets).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


import json


def main(config):
    set_seed(config)
    device = get_device(config)

    root = config["data"]["root"]
    batch_size = config["training"]["batch_size"]

    train_loader = get_loader(root=root, batch_size=batch_size, split="train")
    val_loader = get_loader(root=root, batch_size=batch_size, split="val")

    model = models.resnet50(pretrained=True)

    model.to(device)

    criterion = get_criterion(config)
    optimizer = get_optimizer(model, config)

    metrics = []

    metrics.append(torchmetrics.Accuracy(**{}))

    criterion = get_criterion(config)
    optimizer = get_optimizer(model, config)

    epochs = config["training"]["epochs"]
    history = {"epochs": epochs, "training": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_details = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history["training"].append(epoch_details)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

    # save final results explicitly clearly
    import json

    Path("results.json").write_text(json.dumps(history, indent=4))
    print(f"Results explicitly saved at: {Path('results.json').resolve()}")
