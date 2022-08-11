import torch
import torch.nn as nn
import torch.nn.functional as F


class RepConv(nn.Module):

    def __init__(self, channels, kernel_sizes=(3, 7, 11), dilation=1):
        super(RepConv, self).__init__()

        self.channels = channels
        self.dilation = dilation
        self.kernel_sizes = sorted(kernel_sizes)
        self.max_kernel_size = max(self.kernel_sizes)
        self.padding = (self.max_kernel_size - 1) // 2 * dilation

        self.convs = nn.ModuleList()
        for k in self.kernel_sizes:
            self.convs.append(nn.Conv1d(channels, channels, k,
                                        dilation=dilation,
                                        padding=(k - 1) // 2 * dilation))

    def forward(self, x):
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        return x + sum(conv_outputs) / len(self.convs)

    def inference(self, x):
        if not hasattr(self, 'weight') or not hasattr(self, 'bias'):
            raise ValueError('do not has such attribute, please call _convert_weight_and_bias first!')

        return F.conv1d(x, self.weight.to(x.device), self.bias.to(x.device), dilation=self.dilation,
                        padding=self.padding)

    def convert_weight_bias(self):
        weight = self.convs[-1].weight
        bias = self.convs[-1].bias

        # add other conv
        for conv in self.convs[:-1]:
            pad = (self.max_kernel_size - conv.weight.shape[-1]) // 2
            weight = weight + F.pad(conv.weight, [pad, pad])
            bias = bias + conv.bias

        weight = weight / len(self.convs)
        bias = bias / len(self.convs)

        # add identity
        pad = (self.max_kernel_size - 1) // 2
        weight = weight + F.pad(torch.eye(self.channels).unsqueeze(-1).to(weight.device), [pad, pad])

        self.weight = weight.detach()
        self.bias = bias.detach()