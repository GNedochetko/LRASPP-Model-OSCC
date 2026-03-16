import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import ORCADataset
from .model import get_model


# Melhores hiperparametros identificados pelo grid search 
def train_model(
    data_dir,
    num_classes=2,
    batch_size=4,
    epochs=100,
    lr=0.0003,
    weight_decay=1e-05,
    pretrained=True,
    aux_loss=True,
    aux_weight=0.4,
    device="cuda",
):
    train_dataset = ORCADataset(data_dir, subset="training")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = get_model(num_classes=num_classes, pretrained=pretrained)
    if not aux_loss and hasattr(model, "aux_classifier"):
        model.aux_classifier = None  # desativa ramo auxiliar quando nao queremos a loss extra
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)          # dict: out (e possivelmente aux)
            main_loss = criterion(outputs["out"], masks)
            if aux_loss and "aux" in outputs:
                aux_loss_val = criterion(outputs["aux"], masks)
                loss = (1 - aux_weight) * main_loss + aux_weight * aux_loss_val
            else:
                loss = main_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{epochs}]  loss={epoch_loss:.4f}")

    return model


if __name__ == "__main__":
    data_dir = "data/ORCA_512x512"  # caminho para o dataset 
    trained_model = train_model(
        data_dir,
        epochs=100,
        batch_size=4,
        lr=0.0003,
        weight_decay=1e-05,
        pretrained=True,
        aux_loss=True,
        aux_weight=0.4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
