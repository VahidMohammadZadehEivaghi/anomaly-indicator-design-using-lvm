from models import Decomposition
from utils import DecompositionLoss, train_for_one_epoch, evaluation
import torch
from torch.utils.data import DataLoader
import json


input_dim = 16
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    "input_dim": input_dim,
    "batch_size": batch_size,
    "x_m": 2,
    "x_s": 1,
    "lr": 0.001,
    "epochs": 10,
    "l1": 1,
    "l2": 1,
    "l3": 1,
    "l4": 1,
    "l5": 1,
    "device": str(device)
}

noise_extractor_shape = (16, 35, 16)
filter_shape = (16, 35, 16)

samples = 1000 * batch_size
train_data = 2 + torch.randn((samples, input_dim), device=device)
val_data = 2 + torch.randn((int(samples/100), input_dim), device=device)

train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = Decomposition(
    noise_extractor_shape=noise_extractor_shape, filter_shape=filter_shape
)

history = {
    "train_loss": [],
    "val_loss": []
}

model.to(device)
model.train()
epochs = params["epochs"]
for epoch in range(1, epochs + 1):
    print(f"--------------------{epoch} / {epochs}-------------------")
    train_loss = train_for_one_epoch(
        model=model, train_loader=train_loader, params=params
    )
    history["train_loss"].append(train_loss)
    val_loss = evaluation(
        model=model, val_loader=val_loader, params=params
    )
    history["val_loss"].append(val_loss)

with open("history.json", "w") as hist:
    json.dump(history, hist, indent=4)