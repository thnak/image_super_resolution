import torch
from torch import nn

from utils.general import fix_problem_with_reuse_activation_funtion, ACT_LIST, autopad


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation, dropout"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act: any = True, dropout=0.):
        super(Conv, self).__init__()
        if isinstance(d, ACT_LIST):  # Try to be compatible with models from other repo
            act = d
            d = 1
        act = fix_problem_with_reuse_activation_funtion(act)
        assert 0 <= dropout <= 1, f"dropout rate must be 0 <= dropout <= 1, your {dropout}"
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class residual_block_1(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int, kernel: any, act: any):
        super(residual_block_1, self).__init__()
        act = fix_problem_with_reuse_activation_funtion(act)
        self.m = nn.Sequential(Conv(in_channel, hidden_channel, kernel, 1, None, act=act),
                               Conv(hidden_channel, hidden_channel, 3, 1, None, act=act),
                               Conv(hidden_channel, out_channel, 1, 1, None, act=False))
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, inputs: torch.Tensor):
        return self.act(inputs + self.m(inputs))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = Conv(3, 64, 9, 1, None, act=nn.LeakyReLU())
        residual = [Conv(64, 64, 3, 1, None, act=nn.LeakyReLU())] * 16
        self.residual = nn.Sequential(*residual)
        self.subpixel_convolutional_blocks0 = Conv(64, 256, 3, 1, None, act=False)
        self.shuffle0 = nn.PixelShuffle(2)
        self.subpixel_convolutional_blocks1 = Conv(64, 256, 3, 1, None, act=False)
        self.shuffle1 = nn.PixelShuffle(2)
        self.conv1 = Conv(64, 3, 9, 1, None, act=nn.Tanh())

    def forward(self, inputs: torch.Tensor):
        inputs = self.conv0(inputs)
        inputs = inputs + self.residual(inputs)
        inputs = self.subpixel_convolutional_blocks0(inputs)
        inputs = self.shuffle0(inputs)
        inputs = self.subpixel_convolutional_blocks1(inputs)
        inputs = self.shuffle1(inputs)
        inputs = self.conv1(inputs)
        return inputs


if __name__ == '__main__':
    model = Model()
    model.eval()
    feed = torch.zeros([1, 3, 96, 96])
    feed = model(feed)
    print(feed.shape)
