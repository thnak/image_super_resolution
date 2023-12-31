import torch
from torch import nn
from utils.models import TruncatedVGG19


class gen_loss:
    def __init__(self, vgg_i=5, vgg_j=4, beta=1e-3, device='cuda', beforeAct=False):
        self.vgg_net = TruncatedVGG19(vgg_i, vgg_j, beforeAct).to(device)
        for x in self.vgg_net.parameters():
            x.requires_grad = False
        self.vgg_net.eval()
        self.mse = L1Loss() if beforeAct else nn.MSELoss()
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


class L1Loss(nn.Module):
    def __init__(self, lossweight=1):
        super().__init__()
        self.loss_weight = nn.parameter.Parameter(torch.tensor([lossweight]))
        self.criterion = nn.L1Loss()

    def forward(self, inputs, ground_truth):
        return torch.sum(torch.mul(self.loss_weight, self.criterion(inputs, ground_truth)))
