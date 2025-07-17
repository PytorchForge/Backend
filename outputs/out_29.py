import json
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

transforms = []


def get_loader(root, batch_size, split):
    class CustomImageDataset(Dataset):
        def __init__(self, root_dir, split, transform=None):
            self.root_dir = f"{root_dir}/{split}"
            self.transform = transform
            self.image_paths = list(Path(self.root_dir).rglob("*.jpg")) + list(
                Path(self.root_dir).rglob("*.png")
            )
            self.classes = sorted({p.parent.name for p in self.image_paths})
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            label = self.class_to_idx[image_path.parent.name]
            if self.transform:
                image = self.transform(image)
            return {"image": image, "label": label}

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = CustomImageDataset(root, split, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
    return loader


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


def main():
    with open("config.json") as f:
        config = json.load(f)

    set_seed(config)
    device = get_device(config)

    root = config["data"].get("root", ".")
    batch_size = config["training"]["batch_size"]

    train_loader = get_loader(root=root, batch_size=batch_size, split="train")
    val_loader = get_loader(root=root, batch_size=batch_size, split="val")

    model = models.resnet50(pretrained=True)

    model.to(device)

    criterion = get_criterion(config)
    optimizer = get_optimizer(model, config)
    metrics = []

    metrics.append(torchmetrics.Accuracy())

    epochs = config["training"]["epochs"]

    history = {"epochs": epochs, "training": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        history["training"].append(epoch_log)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

    Path("results.json").write_text(json.dumps(history, indent=4))

    print("Training completed. Results are saved in 'results.json'.")


if __name__ == "__main__":
    main()
