"""unet.py

Implements the modified U-Net architecture described in [1]. The code is based
heavily on [2], with slight modifications made to match the feature map sizes
in [1] and facilitate model viewing in Tensorboard.

# [1] K. H. Jin, M. T. McCann, E. Froustey, and M. Unser, “Deep
# convolutional neural network for inverse problems in imaging,” IEEE
# Transactions on Image Processing, vol. 26, no. 9, pp. 4509–4522, 2017.

# [2] https://github.com/amaarora,
# “U-net: A pytorch implementation in 60 lines of code.”
# https://amaarora.github.io/2020/09/13/unet.html, 2020.
# Accessed 27 April 2022.
"""

# Standard Imports
import typing

# Third-Party Imports
import torch
from torch import nn


class DbleConvBlock(nn.Module):
    """Smallest building block of UNet: (Conv2d + BN + ReLU)x2

    Parameters
    ----------
    in_chan : int
        The # of input channels to the 1st convolution
    out_chan : int
        The # of output channels to the 1st, 2nd convolutions
    """

    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()

        # 3x3 conv - zero pad to preserve size (see [1]), bias in BN layer
        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of computations through DbleConvBlock

        Parameters
        ----------
        x : torch.Tensor
            input tensor to the block

        Returns
        ----------
        torch.Tensor
            output tensor from the block
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class ContractPath(nn.Module):
    """The contracting/encoder side of U-net.

    Parameters
    ----------
    chs : (int, int, int, int, int, int, int)
        The number of feature maps at each encoder stage
    """

    def __init__(self,
                 chs: typing.Tuple[int] = (1, 64, 64, 128, 256, 512, 1024)):
        super().__init__()

        # Add extra conv at beginning to match [1]
        self.conv1 = nn.Conv2d(chs[0], chs[1], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(chs[1])
        self.relu1 = nn.ReLU(inplace=True)

        # Contraction path consists of DoubleConvBlocks & MaxPool layers
        self.contract_blocks = nn.ModuleList(
            [DbleConvBlock(chs[i], chs[i+1]) for i in range(1, len(chs)-1)])
        self.pool_list = nn.ModuleList(
            [nn.MaxPool2d(2) for i in range(1, len(chs)-1)])

    def forward(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        """Forward pass of computations through the ContractPath

        Parameters
        ----------
        x : torch.Tensor
            input tensor to the block

        Returns
        -------
        features : [torch.Tensor]*6
            The feature maps for each skip connection in U-net, starting with
            the low-channel, high-spatial dimension group of maps and moving to
            the high-channel, low-spatial dimension group of maps.  To access
            feature maps at skip connection n, simply index using features[n]
        """

        # Pass through initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Save features for expansion path
        features = []

        # Pass input through respective DoubleConvBlocks & MaxPool layers
        for idx, contract_block in enumerate(self.contract_blocks):
            x = contract_block(x)
            features.append(x)
            x = self.pool_list[idx](x)

        return features


class ExpandPath(nn.Module):
    """The expanding/decoder side of U-Net.

    Parameters
    ----------
    chs : (int, int, int, int, int)
        The number of feature maps at each decoder stage
    """

    def __init__(self, chs: typing.Tuple[int] = (1024, 512, 256, 128, 64)):
        super().__init__()
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2)
             for i in range(len(chs)-1)])
        self.expand_blocks = nn.ModuleList(
            [DbleConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x: torch.Tensor,
                encoder_features: typing.List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of computations through the ExpandPath

        Parameters
        ----------
        x : torch.Tensor
            input tensor to the block
        encoder_features : [torch.Tensor]*6
            feature maps encoded by the ContractPath

        Returns
        ----------
        torch.Tensor
            output tensor from the block
        """

        for idx, upconv in enumerate(self.upconvs):
            x = upconv(x)
            # TIP: Note that no cropping is necessary, since the feature maps
            # are the same dimensions spatially
            x = torch.cat([x, encoder_features[idx]], dim=1)
            x = self.expand_blocks[idx](x)

        return x


class UNet(nn.Module):
    """The U-Net model.

    Parameters
    ----------
    enc_chs : (int, int, int, int, int, int, int)
        The number of feature maps at each encoder stage
    dec_chs : (int, int, int, int, int)
        The number of feature maps at each decoder stage
    n_class : int
        Number of output channels for the network
    """

    def __init__(self,
                 enc_chs: typing.Tuple[int] = (1, 64, 64, 128, 256, 512, 1024),
                 dec_chs: typing.Tuple[int] = (1024, 512, 256, 128, 64),
                 n_class: int = 1):

        super().__init__()
        self.encoder = ContractPath(enc_chs)
        self.decoder = ExpandPath(dec_chs)
        self.chan_combine = nn.Conv2d(dec_chs[-1], n_class, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of computations through the U-Net model

        Parameters
        ----------
        x : torch.Tensor
            input tensor to the model

        Returns
        ----------
        torch.Tensor
            model output
        """

        residual = x
        enc_features = self.encoder(x)
        # Reverse the order of the features list, use 0 as x, others as skips
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        out = self.chan_combine(out)

        out += residual

        return out


if __name__ == '__main__':
    unet = UNet()
    model_input = torch.randn(4, 1, 512, 512)

    print(f'Model Input Size: {model_input.shape}')
    print(f'Model Output Size: {unet(model_input).shape}')
