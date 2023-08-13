import torch
import numpy as np
from pathlib import Path
from torch.nn import MSELoss
from torch.amp import autocast
from torch.optim import Adam
import json
from tqdm import tqdm
from utils.models import Model
from utils.datasets import SR_dataset, init_dataloader


def train(model: Model, dataloader, compute_loss: MSELoss, optimizer: Adam, epoch: int):
    model.train()
    losses = []
    device = next(model.parameters()).device

    pbar = tqdm(dataloader, total=len(dataloader))
    for idx, (hr_images, lr_images) in enumerate(pbar):
        optimizer.zero_grad()
        hr_images, lr_images = hr_images.to(device), lr_images.to(device)
        with autocast(device_type=device.type,
                      enabled=device.type == 'cuda'):
            preds = model(lr_images)
        loss = compute_loss(preds, hr_images)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.desc = f"Epoch [{epoch}] Loss: {np.mean(losses)}"

    return np.mean(losses)


if __name__ == '__main__':
    model = Model()
    json_file = Path("/train_images.json")
    lr = 1e-4
    epochs = 300
    dataset = SR_dataset(json_file, 96, 4, "Train: ")
    dataloader = init_dataloader(dataset, batch_size=318)[0]
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr)
    compute_loss = torch.nn.MSELoss()

    best_fitness = 1000

    for x in range(epochs):
        loss = train(model, dataloader, compute_loss, optimizer, x)
        if best_fitness > loss:
            best_fitness = loss
            torch.save({'model': model.state_dict(), "optimizer": optimizer.state_dict()}, f"best_fitness_{x}.pt")
