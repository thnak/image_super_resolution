import torch
from torch import nn
from utils.models import Convert_tanh_value_norm, TruncatedVGG19


class Content_Loss(nn.Module):
    def __init__(self, mean=None, std=None, vgg_i=5, vgg_j=4, beta=1e-3):
        super().__init__()
        self.tanh_to_norm = Convert_tanh_value_norm()
        self.vgg_net = TruncatedVGG19(vgg_i, vgg_j)
        self.content_loss_compute = nn.MSELoss()
        self.adversarial_loss_compute = nn.BCEWithLogitsLoss()
        self.beta = beta

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, sr_discriminated: torch.Tensor):
        device = next(self.vgg_net.parameters()).device
        sr_imgs_in_vgg_space = self.vgg_net(inputs)
        hr_imgs_in_vgg_space = self.vgg_net(target).detach()

        content_loss = self.content_loss_compute(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = self.adversarial_loss_compute(sr_discriminated,
                                                         torch.ones_like(sr_discriminated, device=device))
        perceptual_loss = content_loss + self.beta * adversarial_loss
        return perceptual_loss, content_loss, adversarial_loss


class Adversarial(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_compute = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return self.loss_compute(inputs, torch.zeros_like(inputs)) + self.loss_compute(targets, torch.ones_like(targets))
