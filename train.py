import torch
import numpy as np
from pathlib import Path

from torch.cuda.amp import GradScaler
from torch.nn import MSELoss
from torch import autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils.models import Model, SRGAN, Discriminator
from utils.datasets import SR_dataset, init_dataloader
from utils.loss import Content_Loss, Adversarial


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def train(model: Model, dataloader, compute_loss: MSELoss, optimizer: Adam, gradscaler: GradScaler, epoch: int):
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
        gradscaler.scale(loss).backward()
        gradscaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 10)
        gradscaler.step(optimizer)
        gradscaler.update()

        losses.append(loss.item())
        pbar.desc = f"Epoch [{epoch}] Loss: {np.mean(losses)}"
    return np.mean(losses)


def train_srgan(gen_net: SRGAN, dis_net: Discriminator, dataloader, content_loss: Content_Loss, adv_loss: Adversarial,
                optimizer_g: Adam,
                optimizer_d: Adam,
                gradscaler: GradScaler, epoch: int):
    gen_net.train()
    dis_net.train()
    loss_g = []
    loss_d = []
    loss_content = []
    loss_adv = []
    device = next(gen_net.parameters()).device
    pbar = tqdm(dataloader, total=len(dataloader))
    for idx, (hr_images, lr_images) in enumerate(pbar):
        optimizer_g.zero_grad()
        hr_images, lr_images = hr_images.to(device), lr_images.to(device)
        with autocast(device_type=device.type,
                      enabled=device.type == 'cuda'):
            sr_images = gen_net(lr_images)
            sr_discriminated = dis_net(sr_images)
            loss, cont_loss, adversarial_loss = content_loss(sr_images, hr_images, sr_discriminated)
        gradscaler.scale(loss).backward()
        gradscaler.unscale_(optimizer_g)
        clip_grad_norm_(gen_net.parameters(), 10)
        gradscaler.step(optimizer_g)
        gradscaler.update()
        loss_g.append(loss.item())

        optimizer_d.zero_grad()
        with autocast(device_type=device.type,
                      enabled=device.type == 'cuda'):
            sr_discriminated = dis_net(sr_images.detach())
            hr_discriminated = dis_net(hr_images)
            loss = adv_loss(sr_discriminated, hr_discriminated)
        gradscaler.scale(loss).backward()
        gradscaler.unscale_(optimizer_d)
        clip_grad_norm_(gen_net.parameters(), 10)
        gradscaler.step(optimizer_d)
        gradscaler.update()
        loss_d.append(loss.item())
        loss_content.append(cont_loss.item())
        loss_adv.append(adversarial_loss.item())
        pbar.desc = (f"Epoch [{epoch}] Loss gen: {np.mean(loss_g)}, Loss dis: {np.mean(loss_d)}, "
                     f"Content: {np.mean(loss_content)}, Adv: {np.mean(loss_adv)}")

    return 0


if __name__ == '__main__':
    gen_net = SRGAN()
    dis_net = Discriminator(3, 64, 8, 1024)
    json_file = Path("./train_images.json")
    checkpoints = Path("")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    lr = 1e-4
    epochs = 300
    dataset = SR_dataset(json_file, 96, 4, "Train: ")
    dataloader = init_dataloader(dataset, batch_size=64, num_worker=4)[0]
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, gen_net.parameters()),
                                   lr=lr)
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, dis_net.parameters()),
                                   lr=lr)

    scaler = GradScaler(enabled=device.type == 'cuda')

    start_epoch = 0
    if checkpoints.is_file():
        print(f"Train: load state dict from {checkpoints.as_posix()}")
        ckpt = torch.load(checkpoints.as_posix(), "cpu")
        gen_net.load_state_dict(ckpt['gen_net'])
        dis_net.load_state_dict(ckpt["dis_net"])
        optimizer_g.load_state_dict(ckpt['optimizer_g'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        start_epoch = ckpt.get('epoch', 0)

    content_loss_compute = Content_Loss()
    adv_loss_compute = Adversarial()
    gen_net.to(device)
    dis_net.to(device)
    content_loss_compute.to(device).eval()
    adv_loss_compute.to(device).eval()
    optimizer_to(optimizer_g, device)
    optimizer_to(optimizer_d, device)
    best_fitness = 1000

    for x in range(start_epoch, epochs):
        train_srgan(gen_net, dis_net, dataloader, content_loss_compute, adv_loss_compute,
                    optimizer_g, optimizer_d, scaler, x)
        torch.save({'gen_net': gen_net.state_dict(),
                    "dis_net": dis_net.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "epoch": x},
                   f"best_fitness_{x}.pt")
