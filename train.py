from __future__ import annotations

import argparse
import random
import warnings
from copy import deepcopy

import torch
import numpy as np
from pathlib import Path

from torch.cuda.amp import GradScaler
from torch.nn import MSELoss
from torch import nn, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from utils.models import ResNet, EResNet, SRGAN, Discriminator, Denoise, ModelEMA, ConvertTanh2Norm
from utils.general import intersect_dicts
from utils.datasets import SR_dataset, init_dataloader, Noisy_dataset, DeNormalize
from utils.loss import gen_loss
from torch.utils.tensorboard import SummaryWriter


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


def train(model: any, ema: ModelEMA, dataloader, compute_loss: MSELoss, optimizer: any, gradscaler: GradScaler,
          schedule, epoch: int, tensorBoard: SummaryWriter):
    model.train()
    losses = []
    device = next(model.parameters()).device
    ema.ema.to(device)
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    total = len(dataloader)
    pbar = tqdm(dataloader, total=total)
    for idx, (hr_images, lr_images) in enumerate(pbar):
        hr_images, lr_images = hr_images.to(device, non_blocking=True), lr_images.to(device, non_blocking=True)
        for _ in range(1):
            optimizer.zero_grad()
            with autocast(enabled=device.type == 'cuda', device_type=autocast_device):
                preds = model(lr_images)
                loss = compute_loss(preds, hr_images)
            gradscaler.scale(loss).backward()
            gradscaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 10)
            gradscaler.step(optimizer)
            gradscaler.update()
            schedule.step()
            ema.update(model)
            losses.append(loss.item())
            tensorBoard.add_scalar("loss", loss.item(), epoch * total + idx + 1)
        pbar.desc = f"Epoch [{epoch}]..."
    return losses


def train_srgan(gen_net: SRGAN, ema: ModelEMA, dis_net: Discriminator, dataloader,
                compute_loss: gen_loss,
                optimizer_g: Adam | SGD,
                optimizer_d: Adam | SGD,
                gradscaler: tuple[GradScaler, GradScaler],
                schedules: tuple[LinearLR, LinearLR], epoch: int,
                tensorBoard: SummaryWriter):
    gen_net.train()
    dis_net.train()
    loss_g = []
    gradscaler_gen, gradscaler_dis = gradscaler
    schedule_g, schedule_d = schedules
    device = next(gen_net.parameters()).device
    ema.ema.to(device)
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    total = len(dataloader)
    pbar = tqdm(dataloader, total=total)
    mean = dataloader.dataset.mean
    std = dataloader.dataset.std
    mean = torch.tensor(mean, device=device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std = torch.tensor(std, device=device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    for idx, (hr_images, lr_images) in enumerate(pbar):
        hr_images, lr_images = hr_images.to(device, non_blocking=True), lr_images.to(device, non_blocking=True)
        with autocast(device_type=autocast_device,
                      enabled=device.type == 'cuda'):
            sr_images = gen_net(lr_images)
            sr_images = (sr_images + 1.0) / 2.0
            sr_images = (sr_images - mean) / std
            sr_discriminated = dis_net(sr_images)
            perceptual_loss, adversarial_loss_, content_loss = compute_loss.calc_contentLoss(sr_images, hr_images,
                                                                                            sr_discriminated)
        optimizer_g.zero_grad()
        gradscaler_gen.scale(perceptual_loss).backward()
        gradscaler_gen.unscale_(optimizer_g)
        clip_grad_norm_(gen_net.parameters(), 10)
        gradscaler_gen.step(optimizer_g)
        gradscaler_gen.update()
        loss_g.append(content_loss.item())
        tensorBoard.add_scalar("loss/content", content_loss.item(), epoch * total + idx + 1)
        schedule_g.step()
        ema.update(gen_net)
        tensorBoard.add_scalar("loss/adv", adversarial_loss_.item(), epoch * total + idx + 1)

        with autocast(device_type=autocast_device,
                      enabled=device.type == 'cuda'):
            sr_discriminated = dis_net(sr_images.detach())
            hr_discriminated = dis_net(hr_images)
            adversarial_loss = compute_loss.calc_advLoss(sr_discriminated, hr_discriminated)

        optimizer_d.zero_grad()
        gradscaler_dis.scale(adversarial_loss).backward()
        gradscaler_dis.unscale_(optimizer_d)
        clip_grad_norm_(dis_net.parameters(), 10)
        gradscaler_dis.step(optimizer_d)
        gradscaler_dis.update()
        tensorBoard.add_scalar("loss/dis", adversarial_loss.item(), epoch * total + idx + 1)
        schedule_d.step()
        pbar.desc = (f"Epoch [{epoch}]...")

    return loss_g


def first_setup(seed):
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet", action="store_true")
    parser.add_argument("--train_denoise", action="store_true")
    parser.add_argument("--worker", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--work_dir", type=str, default="./")
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--dml", action="store_true")
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--L1_loss", action="store_true")
    parser.add_argument("--rs_deep", type=int, default=16, help="")
    parser.add_argument("--shape", type=int, default=96)
    parser.add_argument("--save_name", type=str, default="checkpoint")
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--add_rate", type=float, default=0.2)
    parser.add_argument("--enchant", action="store_true")
    parser.add_argument("--tpu", action="store_true")

    opt = parser.parse_args()
    first_setup(opt.seed)

    json_file = Path("./train_images.json")
    work_dir = Path(opt.work_dir)
    work_dir.mkdir(exist_ok=True)
    res_checkpoints = Path(f"res_{opt.save_name}_{opt.rs_deep}_{opt.add_rate}.pt")
    gen_checkpoints = Path(f"gen_{opt.save_name}_{opt.rs_deep}_{opt.add_rate}.pt")
    denoise_checkpoints = Path(f"denoise_{opt.save_name}_{opt.rs_deep}_{opt.add_rate}.pt")
    res_checkpoints = work_dir / res_checkpoints
    gen_checkpoints = work_dir / gen_checkpoints
    denoise_checkpoints = work_dir / denoise_checkpoints
    tensorBoard = SummaryWriter(work_dir.as_posix(), comment=opt.save_name, flush_secs=30, max_queue=200)

    if opt.tpu:
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    else:
        if opt.dml:
            import torch_directml

            device = torch_directml.device(0)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = opt.lr
    epochs = opt.epochs
    weight_decay = opt.weight_decay
    momentum = opt.momentum
    start_epoch = 0
    workers = opt.worker
    batch_size = opt.batch_size
    scaler_gen = GradScaler(enabled=device.type == 'cuda')
    scaler_dis = GradScaler(enabled=device.type == "cuda")

    data_mean = None
    data_std = None

    if opt.train_denoise:
        model = Denoise(opt.rs_deep)
        ema = ModelEMA(model)
        model.to(device)
        for x in model.parameters():
            x.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr)
        if denoise_checkpoints.is_file():
            ckpt = torch.load(denoise_checkpoints.as_posix(), 'cpu')
            print(f"load from {denoise_checkpoints.as_posix()}")
            checkpoint_state = intersect_dicts(ckpt['gen_net'].float().state_dict(), model.state_dict())
            model.load_state_dict(checkpoint_state, strict=False)
            if len(checkpoint_state) == len(model.state_dict()):
                if ckpt['optimizer'] is not None:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    start_epoch = ckpt['epoch'] + 1
                    optimizer_to(optimizer, device)
            data_std = ckpt.get('std', None)
            data_mean = ckpt.get('mean', None)
            print(f"Loaded pre-trained {len(checkpoint_state)}/{len(model.state_dict())} model")
            del ckpt, checkpoint_state

        dataset = Noisy_dataset(json_path=json_file.as_posix(), target_size=opt.shape,
                                prefix="Train: ")

        dataloader = init_dataloader(dataset, batch_size=batch_size, num_worker=workers, shuffle=True)[0]
        schedule = LinearLR(optimizer, start_factor=1, end_factor=opt.lr2,
                            total_iters=epochs * len(dataloader))

        compute_loss = nn.MSELoss()
        n_p = sum([x.numel() for x in model.parameters()])
        n_g = sum([x.numel() for x in model.parameters() if x.requires_grad])
        print(f"Model: {n_p:,} parameters, {n_g:,} gradients")
        for epoch in range(start_epoch, epochs):
            train(model, ema, dataloader, compute_loss, optimizer, scaler_gen, schedule, epoch, tensorBoard)
            torch.save({'gen_net': deepcopy(model).cpu().half(),
                        "optimizer": optimizer.state_dict() if epoch != epochs - 1 else None,
                        "epoch": epoch,
                        "mean": dataset.mean,
                        "std": dataset.std}, denoise_checkpoints.as_posix())

    else:
        dataset = SR_dataset(json_file, opt.shape, 4, opt.mean, "Train: ")
        if not opt.resnet:
            dataset = dataset.set_transform_hr()
        dataloader = init_dataloader(dataset, batch_size=batch_size, num_worker=workers)[0]
        if not opt.resume:
            denorm = DeNormalize(mean=dataset.mean, std=dataset.std)
            for idx, (hr, lr) in enumerate(dataloader):
                tensorBoard.add_images("images/hr", denorm(hr), idx)
                tensorBoard.add_images("images/lr", denorm(lr), idx)
                if idx == 10:
                    del hr, lr
                    break
        prefix = "Train: "
        if opt.resnet:
            model = EResNet(opt.rs_deep, opt.add_rate) if opt.enchant else ResNet(opt.rs_deep, opt.add_rate)
            for x in model.parameters():
                x.requires_grad = True
            ema = ModelEMA(model, tau=epochs * len(dataloader))
            tensorBoard.add_graph(model, torch.zeros([2, 3, 96, 96]))
            model.to(device)
            compute_loss = nn.MSELoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999),
                                         weight_decay=weight_decay)

            schedule = LinearLR(optimizer, start_factor=1, end_factor=opt.lr2,
                                total_iters=epochs * len(dataloader))
            start_epoch = 0
            n_P = sum([x.numel() for x in model.parameters()])
            n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
            print(f"{prefix} {epochs} epochs, {n_P:,} parameters, {n_g:,} gradients")
            if opt.resume:
                if res_checkpoints.is_file():
                    ckpt = torch.load(res_checkpoints.as_posix(), 'cpu')
                    checkpoint_state = intersect_dicts(ckpt['ema'].float().state_dict(), model.state_dict())
                    ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                    ema.updates = ckpt['updates']
                    model.load_state_dict(checkpoint_state, strict=False)
                    if len(checkpoint_state) == len(model.state_dict()):
                        if ckpt.get("optimizer") is not None:
                            optimizer.load_state_dict(ckpt['optimizer'])
                        scaler_gen.load_state_dict(ckpt['scaler'])
                        start_epoch = ckpt['epoch'] + 1
                        # optimizer_to(optimizer, device)
                    print(f"Loaded pre-trained {len(checkpoint_state)}/{len(model.state_dict())} model")
                    for x in model.parameters():
                        x.requires_grad = True
                    del ckpt, checkpoint_state

            for epoch in range(start_epoch, epochs):
                loss = train(model, ema, dataloader, compute_loss, optimizer, scaler_gen, schedule, epoch, tensorBoard)
                torch.save({"gen_net": deepcopy(model).half(),
                            "optimizer": optimizer.state_dict() if epoch != epochs - 1 else None,
                            "epoch": epoch,
                            "mean": dataset.mean,
                            "std": dataset.std, "loss": loss,
                            "scaler": scaler_gen.state_dict(),
                            "ema": deepcopy(ema.ema).half(),
                            "updates": ema.updates},
                           res_checkpoints.as_posix())

        else:
            gen_net = SRGAN(opt.rs_deep, opt.add_rate, opt.enchant)
            gen_net.init_weight(pretrained=res_checkpoints.as_posix())
            dis_net = Discriminator(3, 64, 8, 1024)
            ema = ModelEMA(gen_net, tau=epochs * len(dataloader))

            for x in gen_net.parameters():
                x.requires_grad = True
            for x in dis_net.parameters():
                x.requires_grad = True

            optimizer_g = torch.optim.Adam(params=gen_net.parameters(), lr=opt.lr, betas=(0.9, 0.999),
                                           weight_decay=weight_decay)
            optimizer_d = torch.optim.Adam(params=dis_net.parameters(), lr=opt.lr, betas=(0.9, 0.999),
                                           weight_decay=weight_decay)
            schedule_g = LinearLR(optimizer_g, start_factor=1, end_factor=opt.lr2,
                                  total_iters=epochs * len(dataloader))
            schedule_d = LinearLR(optimizer_d, start_factor=1, end_factor=opt.lr2,
                                  total_iters=epochs * len(dataloader))

            n_P = sum([x.numel() for x in gen_net.parameters()])
            n_g = sum(x.numel() for x in dis_net.parameters())
            print(f"{prefix} {epochs} epochs, gen {n_P:,} parameters, dis {n_g:,} parameters")

            start_epoch = 0
            if opt.resume:
                if gen_checkpoints.is_file():
                    print(f"Train: load state dict from {gen_checkpoints.as_posix()}")
                    ckpt = torch.load(gen_checkpoints.as_posix(), "cpu")
                    gen_net.load_state_dict(intersect_dicts(ckpt['ema'].float().state_dict(), gen_net.state_dict()),
                                            strict=False)
                    dis_net.load_state_dict(intersect_dicts(ckpt['dis_net'].float().state_dict(), dis_net.state_dict()),
                                            strict=False)
                    if ckpt.get("optimizer_g") is not None:
                        optimizer_g.load_state_dict(ckpt['optimizer_g'])
                        optimizer_d.load_state_dict(ckpt['optimizer_d'])
                        optimizer_to(optimizer_g, device)
                        optimizer_to(optimizer_d, device)
                    scaler_gen.load_state_dict(ckpt['scaler_res'])
                    scaler_dis.load_state_dict(ckpt['scaler_gen'])
                    ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                    ema.updates = ckpt['updates']
                    start_epoch = ckpt['epoch'] + 1
                    for x in gen_net.parameters():
                        x.requires_grad = True
                    for x in dis_net.parameters():
                        x.requires_grad = True

                    del ckpt
                else:
                    if res_checkpoints.is_file():
                        gen_net.res_net.load_state_dict(torch.load(res_checkpoints, "cpu")['gen_net'], strict=False)

            compute_loss = gen_loss(device=device, beforeAct=opt.enchant)

            gen_net.to(device)
            dis_net.to(device)
            n_P = sum([x.numel() for x in gen_net.parameters()])
            n_g = sum(x.numel() for x in gen_net.parameters() if x.requires_grad)  # number gradients
            print(f"{prefix}{n_P:,} parameters, {n_g:,} gradients")

            best_fitness = 1000

            for x in range(start_epoch, epochs):
                loss = train_srgan(gen_net=gen_net, ema=ema,
                                   dis_net=dis_net, dataloader=dataloader,
                                   compute_loss=compute_loss,
                                   optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                                   gradscaler=(scaler_gen, scaler_dis),
                                   schedules=(schedule_g, schedule_d), epoch=x,
                                   tensorBoard=tensorBoard)

                torch.save({'gen_net': deepcopy(gen_net).half(),
                            "dis_net": deepcopy(dis_net).half(),
                            "optimizer_g": optimizer_g.state_dict() if x != epochs - 1 else None,
                            "optimizer_d": optimizer_d.state_dict() if x != epochs - 1 else None,
                            "mean": dataset.mean, "std": dataset.std,
                            "loss": loss,
                            "epoch": x,
                            "scaler_gen": scaler_dis.state_dict(),
                            "scaler_res": scaler_gen.state_dict(),
                            "ema": deepcopy(ema.ema).half(),
                            "updates": ema.updates},
                           gen_checkpoints.as_posix())
            tensorBoard.close()
