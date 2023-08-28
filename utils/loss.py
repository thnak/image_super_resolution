import torch
from torch import nn
from utils.models import Convert_tanh_value_norm, TruncatedVGG19


class Content_Loss:
    def __init__(self, vgg_i=5, vgg_j=4, beta=1e-3, device='cuda'):
        self.vgg_net = TruncatedVGG19(vgg_i, vgg_j).eval().to(device)

        self.content_loss_compute = nn.MSELoss()
        self.adversarial_loss_compute = nn.BCEWithLogitsLoss()
        self.beta = beta

    def __call__(self, inputs: torch.Tensor, target: torch.Tensor, sr_discriminated: torch.Tensor):
        sr_imgs_in_vgg_space = self.vgg_net(inputs)
        hr_imgs_in_vgg_space = self.vgg_net(target).detach()

        content_loss = self.content_loss_compute(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = self.adversarial_loss_compute(sr_discriminated,
                                                         torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + self.beta * adversarial_loss
        return perceptual_loss, content_loss, adversarial_loss


class Adversarial:
    def __init__(self):
        self.loss_compute = nn.BCEWithLogitsLoss()

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor):
        return self.loss_compute(inputs, torch.zeros_like(inputs)) + self.loss_compute(targets,
                                                                                       torch.ones_like(targets))
