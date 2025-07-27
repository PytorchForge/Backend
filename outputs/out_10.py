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
from torchvision import transforms as T

print("Unsupported model type: ")


transforms = [
    T.ToTensor(),
]


class ImageClassificationDataset(Dataset):
    def __init__(
        self, root, split="train", return_format="dict", label_map=None, transforms=None
    ):
        self.root = Path(root) / split
        self.return_format = return_format
        self.transforms = (
            transforms
            if transforms
            else [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

        self.label_map = label_map or self._infer_label_map()
        self.data = self._load_data()

    def _infer_label_map(self):
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        return {cls: idx for idx, cls in enumerate(classes)}

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
        image = Image.open(img_path).convert("RGB")

        for transform in self.transforms:
            image = transform(image)

        if self.return_format == "tuple":
            return image, label
        elif self.return_format == "raw":
            return image
        else:
            return {"image": image, "label": label}


def get_image_classification_loader(
    root, batch_size=32, split="train", shuffle=True, transforms=None, **kwargs
):
    dataset = ImageClassificationDataset(
        root=root, split=split, transforms=transforms, **kwargs
    )
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


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=[0.9, 0.999],
    eps=1e-08,
    weight_decay=0,
    amsgrad=True,
)


criterion = torch.nn.CrossEntropyLoss(
    weight=None,
    size_average=False,
    ignore_index=-100,
    reduce=False,
    reduction="Mean",
    label_smoothing=0,
)


metrics = []

metrics.append(
    torchmetrics.Accuracy(
        task="multiclass", num_classes=2, threshold=0.5, top_k=None, average="none"
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

    model = CustomModel(config)

    model.to(device)

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

    print("Training completed. Results saved in 'results.json'.")


if __name__ == "__main__":
    main()
