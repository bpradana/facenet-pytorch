import math

import torch
import torch.nn as nn
import torch.nn.functional as F


configurations = {
    "embedding_size": 512,
    "input_channel": 3,
    "layers": [
        [  # v, d, e,  o,  se, st
            [1, 3, 16, 16, 0, 1],
            [1, 3, 48, 24, 0, 2],
        ],
        [
            [2, 3, 72, 24, 0, 1],
            [2, 5, 72, 40, 0.25, 2],
        ],
        [
            [2, 5, 120, 40, 0.25, 1],
            [2, 3, 240, 80, 0, 2],
        ],
        [
            [2, 3, 200, 80, 0, 1],
            [2, 3, 184, 80, 0, 1],
            [2, 3, 184, 80, 0, 1],
            [2, 3, 480, 112, 0.25, 1],
        ],
        [
            [2, 3, 672, 112, 0.25, 1],
            [2, 5, 672, 160, 0.25, 2],
        ],
        [
            [2, 5, 960, 160, 0, 1],
            [2, 5, 960, 160, 0.25, 1],
            [2, 5, 960, 160, 0, 1],
            [2, 5, 960, 160, 0.25, 1],
        ],
    ],
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
    ):
        super(ConvBnAct, self).__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.convolution(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels,
        se_ratio=0.25,
        reduced_base_channels=None,
        divisor=4,
    ):
        super(SqueezeExcite, self).__init__()
        reduced_chs = _make_divisible(
            (reduced_base_channels or in_channels) * se_ratio, divisor
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, reduced_chs, kernel_size=1, bias=True)
        self.activation = nn.PReLU()
        self.conv_expand = nn.Conv2d(reduced_chs, in_channels, kernel_size=1, bias=True)
        self.gate_fn = nn.Hardsigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.activation(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class DepthWiseShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, dw_size=3, stride=1):
        super(DepthWiseShortcut, self).__init__()
        self.dw_shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=dw_size,
                stride=stride,
                padding=(dw_size - 1) // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.dw_shortcut(x)


class DFCAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(DFCAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 5),
                stride=1,
                padding=(0, 2),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(5, 1),
                stride=1,
                padding=(2, 0),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return F.interpolate(attention_map, size=x.size()[2:], mode="nearest")


class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        ratio=2,
        dw_size=3,
        with_activation=True,
    ):
        super(GhostModule, self).__init__()

        self.output_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                init_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.PReLU() if with_activation else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                kernel_size=dw_size,
                stride=1,
                padding=dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.PReLU() if with_activation else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.output_channels, :, :]


class GhostBottleneckV1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        dw_size=3,
        stride=1,
        se_ratio=0.25,
    ):
        super(GhostBottleneckV1, self).__init__()
        self.ghost_module_1 = GhostModule(in_channels, hidden_channels)
        self.dw_conv = (
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=dw_size,
                stride=stride,
                padding=dw_size // 2,
                groups=hidden_channels,
                bias=False,
            )
            if stride > 1
            else None
        )
        self.dw_bn = nn.BatchNorm2d(hidden_channels) if stride > 1 else None
        self.se_module = (
            SqueezeExcite(hidden_channels, se_ratio) if se_ratio > 0 else None
        )
        self.ghost_module_2 = GhostModule(
            hidden_channels, out_channels, with_activation=False
        )
        self.dw_shortcut = (
            DepthWiseShortcut(in_channels, out_channels, dw_size=dw_size, stride=stride)
            if stride > 1 or in_channels != out_channels
            else None
        )

    def forward(self, x):
        residual = x
        x = self.ghost_module_1(x)
        if self.dw_conv is not None and self.dw_bn is not None:
            x = self.dw_conv(x)
            x = self.dw_bn(x)
        if self.se_module is not None:
            x = self.se_module(x)
        x = self.ghost_module_2(x)
        if self.dw_shortcut is not None:
            residual = self.dw_shortcut(residual)
        x += residual
        return x


class GhostBottleneckV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        dw_size=3,
        stride=1,
        se_ratio=0.25,
    ):
        super(GhostBottleneckV2, self).__init__()
        self.ghost_module_1 = GhostModule(in_channels, hidden_channels)
        self.dfc_attention = DFCAttention(in_channels, hidden_channels)
        self.dw_conv = (
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=dw_size,
                stride=stride,
                padding=dw_size // 2,
                groups=hidden_channels,
                bias=False,
            )
            if stride > 1
            else None
        )
        self.dw_bn = nn.BatchNorm2d(hidden_channels) if stride > 1 else None
        self.se_module = (
            SqueezeExcite(hidden_channels, se_ratio) if se_ratio > 0 else None
        )
        self.ghost_module_2 = GhostModule(
            hidden_channels, out_channels, with_activation=False
        )
        self.dw_shortcut = (
            DepthWiseShortcut(in_channels, out_channels, dw_size=dw_size, stride=stride)
            if stride > 1 or in_channels != out_channels
            else None
        )

    def forward(self, x):
        residual = x
        ghost = self.ghost_module_1(x)
        attention_map = self.dfc_attention(x)
        x = ghost * attention_map
        if self.dw_conv is not None and self.dw_bn is not None:
            x = self.dw_conv(x)
            x = self.dw_bn(x)
        if self.se_module is not None:
            x = self.se_module(x)
        x = self.ghost_module_2(x)
        if self.dw_shortcut is not None:
            residual = self.dw_shortcut(residual)
        x += residual
        return x


class RecognitionHead(nn.Module):
    def __init__(
        self, in_channels, embedding_size=512, kernel_size=7, num_classes=None
    ):
        super(RecognitionHead, self).__init__()
        self.embedding_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, embedding_size, kernel_size=1, stride=1, bias=False),
            nn.Flatten(),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, num_classes if num_classes else embedding_size),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.embedding_head(x)


class GhostNet(nn.Module):
    def __init__(self, in_channels, configs, embedding_size, alpha=1.3):
        super(GhostNet, self).__init__()
        out_channels = _make_divisible(16 * alpha, 4)
        self.conv_1 = ConvBnAct(in_channels, out_channels, kernel_size=3, stride=2)
        self.ghost_bottlenecks = {
            1: GhostBottleneckV1,
            2: GhostBottleneckV2,
        }

        stages = []
        for stage in configs:
            layers = []
            for layer in stage:
                version, dw_size, expansion_size, output_size, se_ratio, stride = layer
                expansion_new = _make_divisible(expansion_size * alpha, 4)
                out_new = _make_divisible(output_size * alpha, 4)
                layers.append(
                    self.ghost_bottlenecks[version](
                        out_channels,
                        out_new,
                        expansion_new,
                        dw_size=dw_size,
                        stride=stride,
                        se_ratio=se_ratio,
                    )
                )
                out_channels = out_new
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)

        self.conv_2 = ConvBnAct(out_channels, expansion_size, kernel_size=1, stride=1)  # type: ignore

        self.embedding_head = RecognitionHead(expansion_size, embedding_size)  # type: ignore

    def forward(self, x):
        x = self.conv_1(x)
        x = self.stages(x)
        x = self.conv_2(x)
        x = self.embedding_head(x)
        return x


def ghostnet(pretrained=False, progress=True, **kwargs):
    """
    Constructs a GhostNet model from
    `"GhostNet: More Features from Cheap Operations" <https://arxiv.org/abs/1911.11907>` _.
    """
    model = GhostNet(
        in_channels=configurations["input_channel"],
        configs=configurations["layers"],
        embedding_size=configurations["embedding_size"],
    )

    if pretrained:
        # TODO: Load pretrained weights
        pass

    return model
