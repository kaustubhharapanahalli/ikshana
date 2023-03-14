# Paper is available here: https://arxiv.org/abs/1512.03385

from typing import Callable, List, Optional

import torch
from torch import Tensor, nn
from torchsummary import summary


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """
    conv3x3: 3x3 convolution with padding.

    Parameters
    ----------
    in_channels : int
        Input channel size.
    out_channels : int
        Output channel size.
    stride : int, optional
        Stride size, by default 1
    groups : int, optional
        Number of connections between input channels and output channels, by
        default 1
    dilation : int, optional
        controls the spacing between the kernel points, by default 1

    Returns
    -------
    nn.Conv2d
        Returns a Convolution 2D of 3x3 function that can be used for the
        implementation of models.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        dilation=dilation,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """
    conv1x1: 1x1 convolution with padding.

    Parameters
    ----------
    in_channels : int
        Input channel size.
    out_channels : int
        Output channel size.
    stride : int, optional
        Stride size, by default 1

    Returns
    -------
    nn.Conv2d
        Returns a Convolution 2D of 1x1 function that can be used for the
        implementation of models.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    BasicBlock: Class to implement ResNet18 and ResNet34 blocks.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        __init__: Initialize the Basic Block of ResNet.

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int, optional
            stride of the convolution process, by default 1
        downsample : Optional[nn.Module], optional
            downsampling based on the definition of resnet, by default None
        groups : int, optional
            Control on the connection between inputs and outputs, by default 1
        base_width : int, optional
            Base width of the resnet block, by default 64
        dilation : int, optional
            Dilation definition for dilation convolution, by default 1
        norm_layer : Optional[Callable[..., nn.Module]], optional
            Normalization layer to be used for data normalization during the
            training process, by default None

        Raises
        ------
        ValueError
            If the groups and base_width do not match, ValueError is raised.
        NotImplementedError
            For ResNet18 and ResNet34, dilation should not be greater than 1,
            if the provided dilation value is greater than 1,
            NotImplementedError is raised.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        self.conv1 = conv3x3(
            in_channels=in_channels, out_channels=out_channels, stride=stride
        )
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(
            in_channels=out_channels, out_channels=out_channels
        )
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    """
    BottleneckBlock: Class to implement ResNet architecture higher than
    ResNet34 networks.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        __init__: Initialize the Basic Block of ResNet.

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int, optional
            stride of the convolution process, by default 1
        downsample : Optional[nn.Module], optional
            downsampling based on the definition of resnet, by default None
        groups : int, optional
            Control on the connection between inputs and outputs, by default 1
        base_width : int, optional
            Base width of the resnet block, by default 64
        dilation : int, optional
            Dilation definition for dilation convolution, by default 1
        norm_layer : Optional[Callable[..., nn.Module]], optional
            Normalization layer to be used for data normalization during the
            training process, by default None
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self, in_channels: int, block, layers: List[int], num_classes: int = 10
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, block: nn.Module, out_channels, blocks, stride: int = 1
    ):
        downsample = None
        if stride != 1 and self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = list()
        layers.append(
            block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_channels = out_channels

        for i in range(1, blocks):
            layers.append(
                block(in_channels=self.in_channels, out_channels=out_channels)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.average_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def ResNet18(num_classes: int, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    )


def ResNet34(num_classes: int, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
    )


def ResNet50(num_classes: int, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
    )


def ResNet101(num_classes: int, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        block=BasicBlock,
        layers=[3, 4, 23, 3],
    )


def ResNet152(num_classes: int, in_channels: int = 3):
    return ResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        block=BasicBlock,
        layers=[3, 4, 36, 3],
    )
