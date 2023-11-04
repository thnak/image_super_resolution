from __future__ import annotations

import math
import sys
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy

sys.path.append("./")
from utils.general import fix_problem_with_reuse_activation_funtion, ACT_LIST, autopad, intersect_dicts
from utils.datasets import Normalize


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model: nn.Module, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # init EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model):
        """Update EMA parameters"""
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()


class FullyConnected(nn.Module):
    store_bn = nn.Identity()

    def __init__(self, in_channel: int, out_channel: int, act: any = False):
        """Linear + BatchNorm + Act
        act = False -> nn.Identity() otherwise nn.Act"""
        super().__init__()
        act = fix_problem_with_reuse_activation_funtion(act)
        if isinstance(act, nn.PReLU):
            if act.num_parameters != 1:
                act = nn.PReLU(out_channel)
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def _forward_impl(self, inputs: torch.Tensor):
        return self.act(self.bn(self.linear(inputs)))

    def forward(self, inputs):
        return self._forward_impl(inputs)

    def fuseforward(self):
        if isinstance(self.bn, nn.BatchNorm1d):
            self.store_bn = self.bn
            self.bn = nn.Identity()

    def defuseforward(self):
        if isinstance(self.bn, nn.Identity):
            self.bn = self.store_bn
            self.store_bn = nn.Identity()


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation, dropout"""
    store_bn = nn.Identity()

    def __init__(self, c1, c2, k: any = 1, s: any = 1, p=None, g=1, d=1, act: any = True, dropout=0.):
        super(Conv, self).__init__()
        if isinstance(d, ACT_LIST):  # Try to be compatible with models from other repo
            act = d
            d = 1
        act = fix_problem_with_reuse_activation_funtion(act)
        if isinstance(act, nn.PReLU):
            if act.num_parameters != 1:
                act = nn.PReLU(c2)
            else:
                act = nn.PReLU()
        assert 0 <= dropout <= 1, f"dropout rate must be 0 <= dropout <= 1, your {dropout}"
        pad = autopad(k, p, d) if isinstance(k, int) else [autopad(k[0], p, d), autopad(k[1], p, d)]
        self.conv = nn.Conv2d(c1, c2, k, s, pad, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def _forward_impl(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))

    def forward(self, x):
        return self._forward_impl(x)

    def fuseforward(self):
        if isinstance(self.bn, nn.BatchNorm2d):
            self.store_bn = self.bn
            self.bn = nn.Identity()

    def defuseforward(self):
        if isinstance(self.bn, nn.Identity):
            self.bn = self.store_bn
            self.store_bn = nn.Identity()


class ConvTranspose(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation, dropout"""
    store_bn = nn.Identity()

    def __init__(self, c1, c2, k: any = 1, s: any = 1, p=None, g=1, d=1, act: any = True, dropout=0.):
        super(ConvTranspose, self).__init__()
        if isinstance(d, ACT_LIST):  # Try to be compatible with models from other repo
            act = d
            d = 1
        act = fix_problem_with_reuse_activation_funtion(act)
        if isinstance(act, nn.PReLU):
            if act.num_parameters != 1:
                act = nn.PReLU(c2)
            else:
                act = nn.PReLU()
        assert 0 <= dropout <= 1, f"dropout rate must be 0 <= dropout <= 1, your {dropout}"
        pad = autopad(k, p, d) if isinstance(k, int) else [autopad(k[0], p, d), autopad(k[1], p, d)]
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, pad, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def _forward_impl(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))

    def forward(self, x):
        return self._forward_impl(x)

    def fuseforward(self):
        if isinstance(self.bn, nn.BatchNorm2d):
            self.store_bn = self.bn
            self.bn = nn.Identity()

    def defuseforward(self):
        if isinstance(self.bn, nn.Identity):
            self.bn = self.store_bn
            self.store_bn = nn.Identity()


class ConvAIPE(Conv):
    """analysis into polynomial elements of Conv"""

    def __init__(self, c1, c2, kernels, stride=1, act=False):
        super().__init__(c1, c2, k=kernels, s=stride, act=act)
        conv = self.conv
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kernels = conv.kernel_size
        stride = conv.stride
        dilation = conv.dilation
        groups = conv.groups
        act = self.act
        self.conv = nn.Sequential(
            Conv(in_channels, out_channels, [kernels[0], 1], 1, None, groups, dilation[0], act=act),
            Conv(out_channels, out_channels, [1, kernels[1]], stride, None, groups, dilation[0], act=act))
        self.bn = nn.Identity()
        self.drop = nn.Identity()
        self.act = nn.Identity()


class ConvWithoutBN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation, dropout"""
    store_bn = nn.Identity()

    def __init__(self, c1, c2, k: any = 1, s: any = 1, p=None, g=1, d=1, act: any = True, dropout=0.):
        super().__init__()
        if isinstance(d, ACT_LIST):  # Try to be compatible with models from other repo
            act = d
            d = 1
        act = fix_problem_with_reuse_activation_funtion(act)
        if isinstance(act, nn.PReLU):
            if act.num_parameters != 1:
                act = nn.PReLU(c2)
            else:
                act = nn.PReLU()
        assert 0 <= dropout <= 1, f"dropout rate must be 0 <= dropout <= 1, your {dropout}"
        pad = autopad(k, p, d) if isinstance(k, int) else [autopad(k[0], p, d), autopad(k[1], p, d)]
        self.conv = nn.Conv2d(c1, c2, k, s, pad, groups=g, dilation=d, bias=True)
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def _forward_impl(self, x):
        return self.drop(self.act(self.conv(x)))

    def forward(self, x):
        return self._forward_impl(x)


class ResidualBlock1(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int, kernel: any, act: any):
        super(ResidualBlock1, self).__init__()
        self.m = nn.Sequential(
            Conv(in_channel, hidden_channel, kernel, 1, None, act=act),
            Conv(hidden_channel, out_channel, kernel, 1, None, act=False))

    def forward(self, inputs: torch.Tensor):
        return inputs + self.m(inputs)


class ResidualBlock2(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int, kernel: any, act: any):
        super(ResidualBlock2, self).__init__()
        self.m = nn.Sequential(Conv(in_channel, hidden_channel, 1, 1, None, act=act),
                               Conv(hidden_channel, hidden_channel, kernel, 1, None, act=act),
                               Conv(hidden_channel, out_channel, 1, 1, None, act=False))
        self.m1 = Conv(in_channel, out_channel, 1, 1, None, act=False)
        act = fix_problem_with_reuse_activation_funtion(act)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, inputs: torch.Tensor):
        return self.act(self.m1(inputs) + self.m(inputs))


class Mixed_7a(nn.Module):
    def __init__(self, in_channels, stride, act):
        super().__init__()
        self.conv_0 = Conv(in_channels, in_channels, 1, 1, None, act=act)
        self.conv_1 = nn.Sequential(Conv(in_channels, in_channels // 3, 1, 1, None, act=act),
                                    Conv(in_channels // 3, in_channels // 3, 3, 1, None, act=act),
                                    Conv(in_channels // 3, in_channels, 3, 1, None, act=act))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, inputs):
        return torch.cat([self.conv_0(inputs), self.conv_1(inputs), self.max_pool(inputs)])


class Mixed_7b(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv_0 = Conv()


class RDB(nn.Module):
    """Residual Dense Block"""

    def __init__(self, in_channel, growth_channel, kernel_size, act, add_rate=0., use_BN=True):
        super().__init__()
        self.add_rate = add_rate

        if use_BN:
            self.conv0 = Conv(in_channel + growth_channel * 0, growth_channel, kernel_size, 1, None, act=act)
            self.conv1 = Conv(in_channel + growth_channel * 1, growth_channel, kernel_size, 1, None, act=act)
            self.conv2 = Conv(in_channel + growth_channel * 2, growth_channel, kernel_size, 1, None, act=act)
            self.conv3 = Conv(in_channel + growth_channel * 3, growth_channel, kernel_size, 1, None, act=act)
            self.conv = Conv(in_channel + growth_channel * 4, in_channel, kernel_size, 1, None, act=False)
        else:
            self.conv0 = ConvWithoutBN(in_channel + growth_channel * 0, growth_channel, kernel_size, 1, None, act=act)
            self.conv1 = ConvWithoutBN(in_channel + growth_channel * 1, growth_channel, kernel_size, 1, None, act=act)
            self.conv2 = ConvWithoutBN(in_channel + growth_channel * 2, growth_channel, kernel_size, 1, None, act=act)
            self.conv3 = ConvWithoutBN(in_channel + growth_channel * 3, growth_channel, kernel_size, 1, None, act=act)
            self.conv = ConvWithoutBN(in_channel + growth_channel * 4, in_channel, kernel_size, 1, None, act=False)

    def forward(self, inputs):
        output0 = self.conv0(inputs)
        output1 = self.conv1(torch.cat([inputs, output0], 1))
        output2 = self.conv2(torch.cat([inputs, output0, output1], 1))
        output3 = self.conv3(torch.cat([inputs, output0, output1, output2], 1))
        output3 = torch.cat([inputs, output0, output1, output2, output3], 1)
        return torch.add(torch.mul(self.conv(output3), self.add_rate), inputs)


class RDB_PixelShuffle(nn.Module):
    """Residual Dense Block"""

    def __init__(self, filter, out_channel, kernel_size, act, add_rate=0.2):
        super().__init__()
        self.add_rate = add_rate
        self.conv0 = Conv(filter + filter * 0, filter, kernel_size, 1, None, act=act)
        self.conv1 = Conv(filter + filter * 1, filter, kernel_size, 1, None, act=act)
        self.conv2 = Conv(filter + filter * 2, filter, kernel_size, 1, None, act=act)
        self.conv3 = Conv(filter + filter * 3, filter, kernel_size, 1, None, act=act)
        self.conv = Conv(filter, out_channel, kernel_size, 1, None, act=False)

    def forward(self, inputs):
        output0 = self.conv0(inputs)
        output1 = self.conv1(torch.cat([inputs, output0], 1))
        output2 = self.conv2(torch.cat([inputs, output0, output1], 1))
        output3 = self.conv3(torch.cat([inputs, output0, output1, output2], 1))
        output3 = torch.cat([output0, output1, output2, output3], 1)
        output3 = F.pixel_shuffle(output3, 2)
        output3 = F.max_pool2d(output3, kernel_size=2, stride=2, padding=0)
        output3 = self.conv(output3)
        return output3 * self.add_rate + inputs


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, filters: int, kernel: any, act: any, add_rate=0.2, use_BN=True):
        super().__init__()
        assert 0 < add_rate <= 1, f"add must be in range (0, 1]"
        hidden_channel = filters // 2
        if isinstance(act, nn.PReLU):
            if act.num_parameters != 1:
                act = nn.PReLU(hidden_channel)
            else:
                act = nn.PReLU()
        forward_net = []
        for _ in range(3):
            forward_net.append(RDB(filters, hidden_channel, kernel, act, add_rate=add_rate, use_BN=use_BN))
        self.net = nn.Sequential(*forward_net)
        self.add_rate = add_rate

    def forward(self, inputs: torch.Tensor):
        return torch.add(torch.mul(self.net(inputs), self.add_rate), inputs)


class elan(nn.Module):
    def __init__(self, in_channels, out_channels, act, dropout=0.):
        super(elan, self).__init__()
        outs = out_channels // 4
        self.drop = nn.Dropout(dropout)
        self.conv0 = Conv(in_channels, outs, 1, 1, None, act=act)
        self.conv1 = Conv(in_channels, outs, 1, 1, None, act=act)
        self.conv2 = Conv(outs, outs, 3, 1, None, act=act)
        self.conv3 = Conv(outs, outs, 3, 1, None, act=act)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        output_list = [self.conv0(inputs), self.conv1(inputs)]
        output_list.append(self.conv2(output_list[1]))
        output_list.append(self.conv3(output_list[2]))
        return torch.cat(output_list, 1)


class Inception(nn.Module):
    def __init__(self, in_channel, out_channel, act=any):
        super().__init__()
        assert out_channel >= 4, f"Inception node must have output channels >= 4, got {out_channel}"

        self.conv1 = Conv(in_channel, out_channel // 4, 1, 1, act=False)

        conv2_1 = Conv(in_channel, out_channel // 4, 1, 1, act=act)
        conv2_2 = Conv(out_channel // 4, out_channel // 4, 5, 1, act=False)
        self.conv2 = nn.Sequential(conv2_1, conv2_2)

        conv3_1 = Conv(in_channel, out_channel // 4, 1, 1, act=act)
        conv3_2 = Conv(out_channel // 4, out_channel // 4, 7, 1, act=False)
        self.conv3 = nn.Sequential(conv3_1, conv3_2)

        self.conv4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                   Conv(in_channel, out_channel // 4, 1, 1, act=False))

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, inputs):
        feed_0 = self.conv1(inputs)
        feed_1 = self.conv2(inputs)
        feed_2 = self.conv3(inputs)
        feed_3 = self.conv4(inputs)
        return self.act(torch.cat([feed_0, feed_1, feed_2, feed_3], dim=1))


def fuse_conv_and_bn(conv, bn):
    """Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/"""
    if isinstance(conv, nn.Conv2d):
        fused_conv = nn.Conv2d(conv.in_channels,
                               conv.out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               groups=conv.groups,
                               bias=True)
    elif isinstance(conv, nn.Conv1d):
        fused_conv = nn.Conv1d(conv.in_channels,
                               conv.out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               groups=conv.groups,
                               bias=True)
    else:
        fused_conv = nn.Conv3d(conv.in_channels,
                               conv.out_channels,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               groups=conv.groups,
                               bias=True)

    fused_conv = fused_conv.to(conv.weight.device)
    fused_conv = fused_conv.requires_grad_(False)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fused_conv


class ConvertTanh2Norm(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        mean = torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        std = torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, inputs: torch.Tensor):
        out = (inputs + 1.) / 2.
        return (out - self.mean) / self.std


class Tanh2PIL(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.transforms.functional import to_pil_image
        self.to_pil_image = to_pil_image

    def forward(self, inputs: torch.Tensor):
        n_dims = inputs.dim()
        inputs = (inputs + 1.) / 2.
        outputs = []
        if n_dims == 4:
            batch = inputs.size(0)
            for x in range(batch):
                outputs.append(self.to_pil_image(inputs[x, ...]))
        elif n_dims == 3:
            outputs.append(self.to_pil_image(inputs))
        else:
            raise f"only support 3 & 4 dimension, your {n_dims}"
        return outputs


class TanhToArrayImage(nn.Module):
    def __init__(self, max_pixel_value=255.):
        super().__init__()
        self.register_buffer("max_pixel_value", torch.tensor(max_pixel_value))

    def forward(self, inputs):
        inputs = (inputs + 1.) / 2.
        inputs *= self.max_pixel_value
        return inputs.round().to(torch.uint8)


class TruncatedVGG19(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.

    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    """

    def __init__(self, i, j, beforeActivation=True):
        """
        :param i: the index i in the definition above
        :param j: the index j in the definition above
        """
        super(TruncatedVGG19, self).__init__()

        # Load the pre-trained VGG19 available in torchvision
        from torchvision.models import VGG19_Weights
        vgg19 = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        if not (maxpool_counter == i - 1 and conv_counter == j):
            raise "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (i, j)

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        pad = 0 if beforeActivation else 1
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + pad])
        for x in self.modules():
            if hasattr(x, "inplace"):
                x.inplace = True

    def forward(self, inputs: torch.Tensor):
        """
        Forward propagation
        :param inputs: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(inputs)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output


class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        """
        A series of convolutional blocks
        The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        The first convolutional block is unique because it does not employ batch normalization
        :param kernel_size: kernel size in all convolutional blocks
        :param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
        :param n_blocks: number of convolutional blocks
        :param fc_size: size of the first fully connected layer
        """
        super(Discriminator, self).__init__()
        in_channels = 3
        conv_blocks = []
        out_channels = 0
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            if i == 0:
                conv_blocks.append(ConvWithoutBN(in_channels, out_channels,
                                                 kernel_size, 1 if i % 2 == 0 else 2, None,
                                                 act=nn.LeakyReLU(0.2)))
            else:
                conv_blocks.append(Conv(in_channels, out_channels,
                                        kernel_size, 1 if i % 2 == 0 else 2, None,
                                        act=nn.LeakyReLU(0.2)))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Sequential(nn.Linear(out_channels * 6 ** 2, fc_size), nn.LeakyReLU(0.2))
        # self.fc1 = FullyConnected(out_channels * 2**2, fc_size, nn.LeakyReLU(0.2))
        self.fc2 = nn.Linear(fc_size, 1)

        for x in self.modules():
            if hasattr(x, "inplace"):
                x.inplace = True

    def forward(self, inputs):
        """
        Forward propagation.

        :param inputs: high-resolution or super-resolution images which must be classified as
        such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = inputs.size(0)
        output = self.conv_blocks(inputs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        return self.fc2(output)


class Scaler(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor, kernel_size, act):
        super().__init__()
        act = fix_problem_with_reuse_activation_funtion(act)
        out_channel = out_channel * (scale_factor ** 2)
        if isinstance(act, nn.PReLU):
            if act.num_parameters != 1:
                act = nn.PReLU(out_channel // (scale_factor * 2))
        scaler = [ConvWithoutBN(in_channel, out_channel, kernel_size, 1, None, act=False),
                  nn.PixelShuffle(scale_factor), act]

        self.net = nn.Sequential(*scaler)

    def forward(self, inputs: torch.Tensor):
        return self._forward_impl(inputs)

    def _forward_impl(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, num_block_resnet=16, add_rate=0.2):
        super().__init__()

        self.conv0 = ConvWithoutBN(3, 64, 9, 1, None, act=nn.LeakyReLU(0.2))
        residual = [RRDB(64, 3,
                         act=nn.LeakyReLU(0.2), add_rate=add_rate) for _ in range(num_block_resnet)]
        # residual = [ResidualBlock1(64, 64, 64, 3, nn.PReLU(2)) for _ in range(num_block_resnet)]
        self.residual = nn.Sequential(*residual)

        self.conv1 = Conv(64, 64, 3, 1, None, act=False)
        scaler = [Scaler(64, 64,
                         2, 3,
                         nn.LeakyReLU(0.2)) for _ in range(2)]
        self.scaler = nn.Sequential(*scaler)
        self.conv2 = ConvWithoutBN(64, 3, 9, 1, act=nn.Tanh())

        for x in self.modules():
            if hasattr(x, "inplace"):
                x.inplace = True

    def forward(self, inputs: torch.Tensor):
        inputs = self.conv0(inputs)
        inputs = inputs + self.conv1(self.residual(inputs))
        inputs = self.scaler(inputs)
        inputs = self.conv2(inputs)
        return inputs


class EResNet(nn.Module):
    def __init__(self, num_block_resnet=16, add_rate=0.2):
        super().__init__()

        self.conv0 = ConvWithoutBN(3, 64, 9, 1, None, act=nn.PReLU(2))
        residual = [RRDB(64, 3,
                         act=nn.PReLU(2), add_rate=add_rate, use_BN=False) for _ in range(num_block_resnet)]
        # residual = [ResidualBlock1(64, 64, 64, 3, nn.PReLU(2)) for _ in range(num_block_resnet)]
        self.residual = nn.Sequential(*residual)

        self.conv1 = ConvWithoutBN(64, 64, 3, 1, None, act=False)
        scaler = [Scaler(64, 64,
                         2, 3,
                         nn.PReLU(2)) for _ in range(2)]
        self.scaler = nn.Sequential(*scaler)
        self.conv2 = ConvWithoutBN(64, 3, 9, 1, act=nn.Tanh())

        for x in self.modules():
            if isinstance(x, nn.Conv2d):
                x.weight.data *= 0.2

            if hasattr(x, "inplace"):
                x.inplace = True

    def forward(self, inputs: torch.Tensor):
        inputs = self.conv0(inputs)
        inputs = inputs + self.conv1(self.residual(inputs))
        inputs = self.scaler(inputs)
        inputs = self.conv2(inputs)
        return inputs


class SRGAN(nn.Module):

    def __init__(self, deep, add_rate, enchant=False):
        super().__init__()
        self.res_net = EResNet(deep, add_rate) if enchant else ResNet(deep, add_rate)

    def init_weight(self, pretrained):
        try:
            ckpt = torch.load(pretrained, "cpu")
            self.res_net.load_state_dict(ckpt['ema'].float().state_dict())
        except:
            print(f"Could not load Res checkpoint.")

    def forward(self, inputs: torch.Tensor):
        inputs = self.res_net(inputs)
        return inputs


class Denoise(nn.Module):
    def __init__(self, residual_blocks):
        super().__init__()

        self.conv0 = nn.Sequential(ConvWithoutBN(3, 64, 9, 1, act=nn.LeakyReLU(0.2)))
        self.residual_0 = nn.Sequential(*[ResidualBlock1(64, 64,
                                                         64, 3,
                                                         act=nn.LeakyReLU(0.2)) for _ in range(residual_blocks // 2)])
        self.residual_conv0 = ConvWithoutBN(64, 256, 3, 2, act=nn.LeakyReLU(0.2))
        self.residual_1 = nn.Sequential(*[ResidualBlock1(256,
                                                         256,
                                                         256, 3,
                                                         nn.LeakyReLU(0.2)) for _ in range(2)])
        self.residual_conv1 = nn.Sequential(*[nn.PixelShuffle(2), nn.LeakyReLU(0.2)])

        self.residual_2 = nn.Sequential(*[ResidualBlock1(64, 64,
                                                         64, 3,
                                                         act=nn.LeakyReLU(0.2)) for _ in range(residual_blocks // 2)])
        self.conv1 = Conv(64, 64, 3, 1, None, act=False)
        self.conv2 = nn.Sequential(ConvWithoutBN(64, 3, 9, 1, None, act=nn.Tanh()))

        for x in self.modules():
            if hasattr(x, "inplace"):
                x.inplace = True

    def forward(self, inputs: torch.Tensor):
        inputs = self.conv0(inputs)
        residual = self.residual_0(inputs)
        residual = self.residual_conv0(residual)
        residual = self.residual_1(residual)
        residual = self.residual_conv1(residual)
        residual = self.residual_2(residual)
        inputs = inputs + self.conv1(residual)
        inputs = self.conv2(inputs)
        return inputs


def sliding_window(image: torch.Tensor, step: int | list[int, int] | tuple[int, int], windowSize=None):
    """accept ...CHW"""
    if windowSize is None:
        windowSize = step
    if isinstance(step, int):
        step = [step] * 2
    step[0] = min(image.shape[-2], step[0])
    step[1] = min(image.shape[-1], step[1])

    for y in range(0, image.shape[-2], step[0]):
        for x in range(0, image.shape[-1], step[1]):
            yield step, x, y, image[..., y:y + windowSize, x:x + windowSize]


class Model(nn.Module):
    mean = None
    std = None

    def __init__(self, model: callable):
        super().__init__()
        self.net = model

    def init_normalize(self, mean, std):
        self.net = nn.Sequential(Normalize(mean, std, dim=4), self.net, TanhToArrayImage())

    def init_windowslice(self, feed):

        pass

    def forward(self, inputs: torch.Tensor):
        return self.net(inputs)

    def fuse(self):
        """fuse model Conv2d() + BatchNorm2d() layers, fuse Conv2d + im"""
        prefix = "Fusing layers... "
        p_bar = tqdm(self.modules(), desc=f'{prefix}', unit=" layer")
        for m in p_bar:
            p_bar.set_description_str(f"fusing {m.__class__.__name__}")
            if isinstance(m, Conv):
                if hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    m.fuseforward()
        return self

    def defuse(self):
        """de-fuse model Conv2d() + BatchNorm2d() layers, fuse Conv2d + im"""
        prefix = "Fusing layers... "
        p_bar = tqdm(self.modules(), desc=f'{prefix}', unit=" layer")
        for m in p_bar:
            p_bar.set_description_str(f"fusing {m.__class__.__name__}")
            if isinstance(m, Conv):
                m.defuseforward()
        return self


if __name__ == '__main__':
    try:
        import torch_directml

        device = torch_directml.device(0)
    except Exception as ex:
        device = torch.device(0) if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = Model(SRGAN(23, 0.2, enchant=False))

    # /content/drive/MyDrive/Colab Notebooks/res_checkpoint.pt
    ckpt = torch.load("../gen_RRDB_23_0.2.pt", "cpu")
    loss = ckpt['loss']
    import numpy as np

    print(np.mean(loss))
    model.net.load_state_dict(ckpt['ema'].float().state_dict())
    model.init_normalize(ckpt['mean'], ckpt['std'])
    for x in model.parameters():
        x.requires_grad = False
    model.eval().fuse()

    feed = torch.zeros([1, 3, 96, 96], dtype=torch.uint8, device=device)
    model.to(device)

    if torch.cuda.is_available():
        model.half()
        feed = feed.half()

    n_p = sum([x.numel() for x in model.parameters()])
    print(f"{n_p:,}")
    from time import perf_counter

    t0 = perf_counter()
    for x in range(1):
        model(feed)
    print(f"times: {perf_counter() - t0}")
    jit_m = torch.jit.trace(model, feed)
    torch.jit.save(jit_m, "model.pt")
    axe = {'images': {2: "x", 3: "x"}, "outputs": {}}
    # axe = None
    torch.onnx.export(model, feed, "model.onnx", dynamic_axes=axe,
                      input_names=["images"], output_names=["outputs"])
    import onnx
    from onnxsim import simplify

    onnx_model = onnx.load('model.onnx')
    onnx_model, c = simplify(onnx_model)
    onnx.save(onnx_model, "model.onnx")
    print(feed.shape)
