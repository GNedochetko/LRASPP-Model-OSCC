import torch
from src.train import train_model
from src.test import test_model

DATA_DIR = "data/ORCA_512x512"
MODEL_PATH = "trained_model1.pth"
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_KWARGS = dict(
    data_dir=DATA_DIR,
    num_classes=NUM_CLASSES,
    batch_size=4,
    lr=0.0003,
    weight_decay=1e-05,
    aux_loss=True,
    pretrained=False,
    epochs=100,  
    device=DEVICE,
)


def main():
    #print("Iniciando treinamento do modelo...")
    #model = train_model(**TRAIN_KWARGS)
    #torch.save(model.state_dict(), MODEL_PATH)
    #print(f"Modelo salvo em '{MODEL_PATH}'")

    print("\nIniciando teste do modelo...")
    test_model(
        MODEL_PATH,
        DATA_DIR,
        num_classes=NUM_CLASSES,
        batch_size=TRAIN_KWARGS["batch_size"],
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
