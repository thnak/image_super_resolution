import torch
from torch import nn
from utils.models import ConvertTanh2Norm, TruncatedVGG19


class gen_loss:
    def __init__(self, vgg_i=5, vgg_j=4, beta=1e-3, device='cuda'):
        self.vgg_net = TruncatedVGG19(vgg_i, vgg_j).to(device)
        for x in self.vgg_net.parameters():
            x.requires_grad = False
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta

    def calc_contentLoss(self, sr_imgs: torch.Tensor, hr_imgs: torch.Tensor, sr_discriminated: torch.Tensor):
        sr_imgs_in_vgg_space = self.vgg_net(sr_imgs)
        hr_imgs_in_vgg_space = self.vgg_net(hr_imgs).detach()

        content_loss = self.mse(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = self.bce(sr_discriminated,
                                    torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + self.beta * adversarial_loss
        return perceptual_loss, adversarial_loss, content_loss

    def calc_advLoss(self, sr_discriminated: torch.Tensor, hr_discriminated: torch.Tensor):
        return self.bce(sr_discriminated,
                        torch.zeros_like(sr_discriminated)) + self.bce(hr_discriminated,
                                                                       torch.ones_like(hr_discriminated))
