import torch
from torch import nn
from utils.models import ConvertTanh2Norm, TruncatedVGG19


class Adversarial:
    def __init__(self):
        self.loss_compute = nn.BCEWithLogitsLoss()

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor):
        return self.loss_compute(inputs, torch.zeros_like(inputs)) + self.loss_compute(targets,
                                                                                       torch.ones_like(targets))


class Content_Loss:
    def __init__(self, vgg_i=5, vgg_j=4, beta=1e-3, mse=True, device='cuda'):
        self.vgg_net = TruncatedVGG19(vgg_i, vgg_j).eval().to(device)
        for x in self.vgg_net.parameters():
            x.requires_grad = False
        self.content_loss_compute = nn.MSELoss() if mse else nn.L1Loss()
        self.beta = beta

    def __call__(self, inputs: torch.Tensor, target: torch.Tensor, sr_discriminated: torch.Tensor, Bce: Adversarial):
        sr_imgs_in_vgg_space = self.vgg_net(inputs)
        hr_imgs_in_vgg_space = self.vgg_net(target).detach()

        content_loss = self.content_loss_compute(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = Bce.loss_compute(sr_discriminated,
                                            torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + self.beta * adversarial_loss
        return perceptual_loss, adversarial_loss


class gen_loss:
    def __init__(self, vgg_i=5, vgg_j=4, beta=1e-3, mse=True, device='cuda'):
        self.vgg_net = TruncatedVGG19(vgg_i, vgg_j).eval().to(device)
        for x in self.vgg_net.parameters():
            x.requires_grad = False
        self.mse = nn.MSELoss() if mse else nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta

    def calc_contentLoss(self, sr_imgs: torch.Tensor, hr_imgs: torch.Tensor, sr_discriminated: torch.Tensor):
        sr_imgs_in_vgg_space = self.vgg_net(sr_imgs)
        hr_imgs_in_vgg_space = self.vgg_net(hr_imgs).detach()

        content_loss = self.mse(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = self.bce(sr_discriminated,
                                    torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + self.beta * adversarial_loss
        return perceptual_loss, adversarial_loss

    def calc_advLoss(self, inputs: torch.Tensor, targets: torch.Tensor):
        return self.bce(inputs, torch.zeros_like(inputs)) + self.bce(targets, torch.ones_like(targets))
