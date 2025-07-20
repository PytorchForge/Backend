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
from torchvision import models
from torchvision import transforms as T

print("Unsupported model type: ResNet18")


transforms = [
    T.Resize(size=[224, 224]),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]


def custom_loader(path):
    # user-defined data loading code
    pass


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


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=[0.9, 0.999],
    eps=1e-08,
    weight_decay=0.0,
    amsgrad=False,
)


criterion = torch.nn.CrossEntropyLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
)


metrics = []

metrics.append(
    torchmetrics.Accuracy(
        task="multiclass", num_classes=2, threshold=None, top_k=None, average="micro"
    )
)


def main():
    with open("config.json") as f:
        config = json.load(f)

    set_seed(config)
    device = get_device(config)

    root = config["data"].get("root", ".")
    batch_size = config["training"]["batch_size"]

    train_loader = custom_loader(root=root, batch_size=batch_size, split="train")
    val_loader = custom_loader(root=root, batch_size=batch_size, split="val")

    model = models.ResNet18(pretrained=True)

    model.to(device)

    criterion = get_criterion(config)
    optimizer = get_optimizer(model, config)
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
