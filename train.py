from __future__ import annotations

import argparse
import warnings
import torch
import numpy as np
from pathlib import Path

from torch.cuda.amp import GradScaler
from torch.nn import MSELoss
from torch import autocast, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, LinearLR
from tqdm import tqdm
from utils.models import ResNet, SRGAN, Discriminator, Denoise, ModelEMA
from utils.general import intersect_dicts
from utils.datasets import SR_dataset, init_dataloader, Noisy_dataset
from utils.loss import gen_loss


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
          schedule, epoch: int):
    model.train()
    losses = []
    device = next(model.parameters()).device
    ema.ema.to(device)

    pbar = tqdm(dataloader, total=len(dataloader))
    for idx, (hr_images, lr_images) in enumerate(pbar):
        hr_images, lr_images = hr_images.to(device), lr_images.to(device)
        for _ in range(1):
            optimizer.zero_grad()
            with autocast(device_type=device.type if device.type == 'cuda' else 'cpu',
                          enabled=device.type == 'cuda'):
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
        pbar.desc = f"Epoch [{epoch}] Loss: {np.mean(losses)} minLoss: {np.min(losses)}"
    return losses


def train_srgan(gen_net: SRGAN, ema: ModelEMA, dis_net: Discriminator, dataloader,
                compute_loss: gen_loss,
                optimizer_g: Adam | SGD,
                optimizer_d: Adam | SGD,
                gradscaler: tuple[GradScaler, GradScaler],
                schedules: tuple, epoch: int):
    gen_net.train()
    dis_net.train()
    loss_g = []
    loss_d = []
    loss_adv = []
    gradscaler_res, gradscaler_gen = gradscaler
    schedule_g, schedule_d = schedules
    device = next(gen_net.parameters()).device
    ema.ema.to(device)
    pbar = tqdm(dataloader, total=len(dataloader))
    for idx, (hr_images, lr_images) in enumerate(pbar):
        hr_images, lr_images = hr_images.to(device), lr_images.to(device)
        for x in range(1):
            for pa in dis_net.parameters():
                pa.requires_grad = False
            with autocast(device_type=device.type,
                          enabled=device.type == 'cuda'):
                sr_images = gen_net(lr_images)
                sr_discriminated = dis_net(sr_images)
                loss, adversarial_loss = compute_loss.calc_contentLoss(sr_images, hr_images, sr_discriminated)
            optimizer_g.zero_grad()
            gradscaler_res.scale(loss).backward()
            gradscaler_res.unscale_(optimizer_g)
            clip_grad_norm_(gen_net.parameters(), 10)
            gradscaler_res.step(optimizer_g)
            gradscaler_res.update()
            loss_g.append(loss.item())
            schedule_g.step()

            for pa in dis_net.parameters():
                pa.requires_grad = True
            with autocast(device_type=device.type,
                          enabled=device.type == 'cuda'):
                sr_discriminated = dis_net(sr_images.detach())
                hr_discriminated = dis_net(hr_images)
                loss = compute_loss.calc_advLoss(sr_discriminated, hr_discriminated)

            optimizer_d.zero_grad()
            gradscaler_gen.scale(loss).backward()
            gradscaler_gen.unscale_(optimizer_d)
            clip_grad_norm_(dis_net.parameters(), 10)
            gradscaler_gen.step(optimizer_d)
            gradscaler_gen.update()
            loss_d.append(loss.item())
            loss_adv.append(adversarial_loss.item())
            schedule_d.step()

        pbar.desc = (
            f"Epoch [{epoch}] Loss gen: {np.mean(loss_g)}, Loss dis: {np.mean(loss_d)}, Adv: {np.mean(loss_adv)}")

    return loss_g


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
    parser.add_argument("--save_name", type=str, default="checkpoint.pt")
    parser.add_argument("--nowarn", action="store_true")
    parser.add_argument("--lr2", type=float, default=0.01)

    opt = parser.parse_args()
    if opt.nowarn:
        warnings.filterwarnings("ignore",
                                message="libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG")
        warnings.warn("libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG")

    json_file = Path("./train_images.json")
    work_dir = Path(opt.work_dir)
    work_dir.mkdir(exist_ok=True)
    res_checkpoints = Path(f"res_{opt.save_name}")
    gen_checkpoints = Path(f"gen_{opt.save_name}")
    denoise_checkpoints = Path("denoise_checkpoint.pt")
    res_checkpoints = work_dir / res_checkpoints
    gen_checkpoints = work_dir / gen_checkpoints
    denoise_checkpoints = work_dir / denoise_checkpoints
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if opt.dml:
        import torch_directml

        device = torch_directml.device(0)
    else:
        device = torch.device(device)
    lr = opt.lr
    epochs = opt.epochs
    weight_decay = opt.weight_decay
    momentum = opt.momentum
    start_epoch = 0
    workers = opt.worker
    batch_size = opt.batch_size
    scaler = GradScaler(enabled=device.type == 'cuda')
    scaler_gan = GradScaler(enabled=device.type == "cuda")

    data_mean = None
    data_std = None

    if opt.train_denoise:
        model = Denoise(opt.rs_deep)
        model.to(device)
        for x in model.parameters():
            x.requires_grad = True
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
        if denoise_checkpoints.is_file():
            ckpt = torch.load(denoise_checkpoints.as_posix(), 'cpu')
            checkpoint_state = intersect_dicts(ckpt['model'], model.state_dict())
            model.load_state_dict(checkpoint_state, strict=False)
            if len(checkpoint_state) == len(model.state_dict()):
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
        compute_loss = nn.MSELoss()
        n_p = sum([x.numel() for x in model.parameters()])
        n_g = sum([x.numel() for x in model.parameters() if x.requires_grad])
        print(f"Model: {n_p:,} parameters, {n_g:,} gradients")
        for epoch in range(start_epoch, epochs):
            train(model, dataloader, compute_loss, optimizer, scaler, epoch)
            torch.save({'model': model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "mean": dataset.mean,
                        "std": dataset.std}, denoise_checkpoints.as_posix())

    else:
        dataset = SR_dataset(json_file, opt.shape, 4, opt.mean, "Train: ")
        if not opt.resnet:
            dataset = dataset.set_transform_hr()
        dataloader = init_dataloader(dataset, batch_size=batch_size, num_worker=workers)[0]
        prefix = "Train: "
        if opt.resnet:
            model = ResNet(opt.rs_deep)
            ema = ModelEMA(model)
            compute_loss = nn.L1Loss() if opt.L1_loss else nn.MSELoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999), eps=0.0001,
                                         weight_decay=weight_decay)

            schedule = LinearLR(optimizer, start_factor=1, end_factor=opt.lr2,
                                total_iters=epochs * len(dataloader))
            start_epoch = 0
            model.to(device)
            n_P = sum([x.numel() for x in model.parameters()])
            n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
            print(f"{prefix} {epochs} epochs, {n_P:,} parameters, {n_g:,} gradients")
            if opt.resume:
                if res_checkpoints.is_file():
                    ckpt = torch.load(res_checkpoints.as_posix(), 'cpu')
                    checkpoint_state = intersect_dicts(ckpt['gen_net'], model.state_dict())
                    ema.ema.load_state_dict(ckpt['ema'])
                    ema.updates = ckpt['updates']
                    model.load_state_dict(checkpoint_state, strict=False)
                    if len(checkpoint_state) == len(model.state_dict()):
                        optimizer.load_state_dict(ckpt['optimizer'])
                        scaler.load_state_dict(ckpt['scaler'])
                        start_epoch = ckpt['epoch'] + 1
                    optimizer_to(optimizer, device)
                    print(f"Loaded pre-trained {len(checkpoint_state)}/{len(model.state_dict())} model")
                    del ckpt, checkpoint_state

            for epoch in range(start_epoch, epochs):
                loss = train(model, ema, dataloader, compute_loss, optimizer, scaler, schedule, epoch)
                torch.save({"gen_net": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "mean": dataset.mean,
                            "std": dataset.std, "loss": loss,
                            "scaler": scaler.state_dict(),
                            "ema": ema.ema.half().state_dict(), "updates": ema.updates},
                           res_checkpoints.as_posix())

        else:
            gen_net = SRGAN(opt.rs_deep)
            ema = ModelEMA(gen_net)
            dis_net = Discriminator(3, 64, 8, 1024)
            for x in gen_net.parameters():
                x.requires_grad = True

            optimizer_g = torch.optim.Adam(params=gen_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=0.0001,
                                           weight_decay=weight_decay)
            optimizer_d = torch.optim.Adam(params=dis_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=0.0001,
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
                    gen_net.load_state_dict(intersect_dicts(ckpt['gen_net'], gen_net.state_dict()))
                    dis_net.load_state_dict(intersect_dicts(ckpt['dis_net'], dis_net.state_dict()))
                    optimizer_g.load_state_dict(ckpt['optimizer_g'])
                    optimizer_d.load_state_dict(ckpt['optimizer_d'])
                    scaler.load_state_dict(ckpt['scaler_res'])
                    scaler_gan.load_state_dict(ckpt['scaler_gen'])
                    ema.ema.load_state_dict(ckpt['ema'])
                    ema.updates = ckpt['updates']
                    start_epoch = ckpt['epoch'] + 1
                    optimizer_to(optimizer_g, device)
                    optimizer_to(optimizer_d, device)
                    del ckpt
                else:
                    if res_checkpoints.is_file():
                        gen_net.net.load_state_dict(torch.load(res_checkpoints, "cpu")['gen_net'])
            else:
                print(f"loaded from pretrained of resnet")
                gen_net.net.load_state_dict(torch.load(res_checkpoints, "cpu")['gen_net'])

            compute_loss = gen_loss(mse=not opt.L1_loss)
            gen_net.to(device)
            dis_net.to(device)
            n_P = sum([x.numel() for x in gen_net.parameters()])
            n_g = sum(x.numel() for x in gen_net.parameters() if x.requires_grad)  # number gradients
            print(f"{prefix}{n_P:,} parameters, {n_g:,} gradients")

            best_fitness = 1000

            for x in range(start_epoch, epochs):
                loss = train_srgan(gen_net, ema, dis_net, dataloader, compute_loss,
                                   optimizer_g, optimizer_d, (scaler, scaler_gan),
                                   (schedule_g, schedule_d), x)

                torch.save({'gen_net': gen_net.state_dict(),
                            "dis_net": dis_net.state_dict(),
                            "optimizer_g": optimizer_g.state_dict(),
                            "optimizer_d": optimizer_d.state_dict(),
                            "mean": dataset.mean, "std": dataset.std,
                            "loss": loss,
                            "epoch": x, "scaler_gen": scaler_gan.state_dict(),
                            "scaler_res": scaler.state_dict(),
                            "ema": ema.ema.half().state_dict(),
                            "updates": ema.updates},
                           gen_checkpoints.as_posix())
