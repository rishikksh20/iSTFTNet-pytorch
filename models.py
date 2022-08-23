import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding

LRELU_SLOPE = 0.1

from rep_conv import RepConv


class Upsample(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

        self.transposed_conv = nn.ConvTranspose1d(in_channels,
                                                  out_channels,
                                                  kernel_size=scale * 2,
                                                  stride=scale,
                                                  padding=scale // 2 + scale % 2,
                                                  output_padding=scale % 2)

    def forward(self, x):
        return self.transposed_conv(x)

    def inference(self, x):
        return self.forward(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 channels: int = 512,
                 kernel_sizes: tuple = (3, 7, 11),
                 dilations: tuple = (1, 3, 5),
                 use_additional_convs: bool = True):
        super(ResidualBlock, self).__init__()

        self.use_additional_convs = use_additional_convs

        self.act = nn.LeakyReLU(0.1)
        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(RepConv(channels, kernel_sizes, dilation=dilation))
            if use_additional_convs:
                self.convs2 += [RepConv(channels, kernel_sizes, dilation=1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            x = self.act(x)
            x = self.convs1[idx](x)
            if self.use_additional_convs:
                x = self.act(x)
                x = self.convs2[idx](x)
        return x

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            x = self.act(x)
            x = self.convs1[idx].inference(x)
            if self.use_additional_convs:
                x = self.act(x)
                x = self.convs2[idx].inference(x)
        return x



class Generator(torch.nn.Module):
    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 channels=512,
                 dropout=0.1,
                 upsample_scales=(8, 8),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilations=((1, 3, 5), (1, 3, 5)),
                 use_additional_convs=True,
                 use_weight_norm=True,
                 training=True
                 ):
        super(Generator, self).__init__()

        # check hyper parameters are valid

        assert len(resblock_dilations) == len(upsample_scales)

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=7, padding=3),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Dropout(dropout)
        )

        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()

        self.non_linear = nn.LeakyReLU(LRELU_SLOPE)
        for i in range(len(upsample_scales)):
            self.upsamples += [
                Upsample(
                    in_channels=channels // (2 ** i),
                    out_channels=channels // (2 ** (i + 1)),
                    scale=upsample_scales[i]
                )
            ]

            self.blocks += [
                ResidualBlock(
                    channels=channels // (2 ** (i + 1)),
                    kernel_sizes=resblock_kernel_sizes,
                    dilations=resblock_dilations[i],
                    use_additional_convs=use_additional_convs
                )
            ]

        self.conv_post = nn.Sequential(
            nn.Conv1d(channels // (2 ** (i + 1)), out_channels + 2, 7, 1, padding=3),
            nn.Tanh()
        )
        self.post_n_fft = out_channels
        self.training = training

        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

    def forward(self, x):
        x = self.input_conv(x)

        for i in range(len(self.upsamples)):
            x = self.upsamples[i](x)
            if self.training:
                x = self.blocks[i](x)
            else:
                x = self.blocks[i].inference(x)
            x = self.non_linear(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

        return spec, phase

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    def convert_weight_bias(self):
        def _convert_weight_bias(m):
            if isinstance(m, RepConv):
                m.convert_weight_bias()

        self.apply(_convert_weight_bias)

    def apply_weight_norm(self):
        """Apply weight normalization module from all the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                nn.utils.weight_norm(m)
            if isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m, dim=1)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):

        def _reset_parameters(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

