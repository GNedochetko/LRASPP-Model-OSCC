import itertools
import os
import csv
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .dataset import ORCADataset
from .model import get_model


def dice_score(preds, masks, num_classes=2):
    # Binary Dice for foreground class 1
    preds = torch.argmax(preds, dim=1)
    preds_fg = (preds == 1).float()
    masks_fg = (masks == 1).float()
    intersection = (preds_fg * masks_fg).sum()
    union = preds_fg.sum() + masks_fg.sum()
    return (2.0 * intersection) / (union + 1e-8)


def train_one_epoch(model, loader, criterion, optimizer, device, use_aux=False, aux_weight=0.4):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        main_loss = criterion(outputs["out"], masks)
        if use_aux and "aux" in outputs:
            aux_loss_val = criterion(outputs["aux"], masks)
            loss = (1 - aux_weight) * main_loss + aux_weight * aux_loss_val
        else:
            loss = main_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / max(len(loader.dataset), 1)


def eval_dice(model, loader, device):
    model.eval()
    total_dice = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)["out"]
            batch_dice = dice_score(outputs, masks)
            total_dice += batch_dice.item() * images.size(0)
            total_samples += images.size(0)
    return total_dice / max(total_samples, 1)


def grid_search_cv(
    data_dir,
    num_classes=2,
    k_folds=5,
    epochs=30,
    device="cuda",
):
    train_dataset = ORCADataset(data_dir, subset="training")

    # Grid definition
    param_grid = {
        "weights": [True, False],
        "aux_loss": [True, False],
        "learning_rate": [1e-4, 3e-4, 1e-5],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "batch_size": [2, 4],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    indices = list(range(len(train_dataset)))
    random.seed(42)
    random.shuffle(indices)
    fold_sizes = [len(indices) // k_folds] * k_folds
    for i in range(len(indices) % k_folds):
        fold_sizes[i] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size

    os.makedirs("resultados", exist_ok=True)
    results_path = os.path.join("resultados", "tune_results.csv")

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys + ["mean_dice"])

        for combo in combos:
            params = dict(zip(keys, combo))
            print(f"Testing params: {params}")

            fold_scores = []

            for fold_idx in range(k_folds):
                val_idx = folds[fold_idx]
                train_idx = [i for f, fold in enumerate(folds) if f != fold_idx for i in fold]
                train_subset = Subset(train_dataset, train_idx)
                val_subset = Subset(train_dataset, val_idx)

                train_loader = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=params["batch_size"], shuffle=False)

                model = get_model(num_classes=num_classes, pretrained=params["weights"]).to(device)
                if not params["aux_loss"] and hasattr(model, "aux_classifier"):
                    model.aux_classifier = None
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=params["learning_rate"],
                    weight_decay=params["weight_decay"],
                )

                for epoch in range(epochs):
                    train_loss = train_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device,
                        use_aux=params["aux_loss"],
                        aux_weight=0.4,
                    )
                    print(
                        f"Fold {fold_idx+1}/{k_folds} - Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}"
                    )

                dice = eval_dice(model, val_loader, device)
                fold_scores.append(dice)
                print(f"Fold {fold_idx+1}/{k_folds} - Dice: {dice:.4f}")

            mean_dice = sum(fold_scores) / max(len(fold_scores), 1)
            writer.writerow([params[k] for k in keys] + [f"{mean_dice:.6f}"])
            print(f"Mean Dice: {mean_dice:.4f}")

    print(f"Grid search finished. Results saved to {results_path}")


if __name__ == "__main__":
    data_dir = "data/ORCA_512x512"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_search_cv(data_dir, k_folds=5, epochs=30, device=device)
