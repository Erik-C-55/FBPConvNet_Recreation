import torch
import torch.nn as nn

# NOTES: This implements the modified U-Net architecture described in [1]. The
#       code is based heavily off on [2], with slight modifications made to 
#       match the feature map sizes in [1] and facilitate model viewing in 
#       Tensorboard.

# [1] K. H. Jin, M. T. McCann, E. Froustey, and M. Unser, “Deep
# convolutional neural network for inverse problems in imaging,” IEEE
# Transactions on Image Processing, vol. 26, no. 9, pp. 4509–4522, 2017.

# [2] https://github.com/amaarora,
# “U-net: A pytorch implementation in 60 lines of code.”
# https://amaarora.github.io/2020/09/13/unet.html, 2020.
# Accessed 27 April 2022.
    
class DbleConvBlock(nn.Module):
    """This class is the smallest building block of UNet - 2 Conv2d layers with
    BN and ReLU between them. Note that BN and ReLU layers have individual
    names to facilitate model viewing in Tensorboard
    
    Parameters:
        in_chan: (int) The # of input channels to the 1st convolution
        out_chan: (int) The # of output channels to the 1st, 2nd convolutions
        
    Returns:
        x: (FloatTensor) The result of the convolutions, BN, and ReLUs
    """
    def __init__(self, in_chan, out_chan):
        super().__init__()
        
        # 3x3 conv - zero pad to preserve size (see [1]), bias in BN layer
        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x
        
class ContractPath(nn.Module):
    """This class consists of the contracting/decoder side of U-net that
    increases the number of feature maps while decreasing spatial dimensions.
    
    Parameters:
        chs: (tuple of ints) The number of feature maps at each encoder stage
        
    Returns:
        features: (list of FloatTensor) The feature maps for each skip
            connection in U-net, starting with the low-channel, high-spatial
            dimension group of maps and moving to the high-channel, low-spatial
            dimension group of maps.  To access feature maps at skip connection
            n, simply index using features[n]
    """
    def __init__(self, chs=(1,64,128,256,512,1024)):
        super().__init__()
        
        # Contraction path consists of DoubleConvBlocks & MaxPool layers
        self.contract_blocks = nn.ModuleList(
            [DbleConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool_list = nn.ModuleList(
            [nn.MaxPool2d(2) for i in range(len(chs)-1)])
        
    def forward(self, x):
        # Save features for expansion path
        features =[]
        
        # Pass input through respective DoubleConvBlocks & MaxPool layers
        for idx in range(len(self.contract_blocks)):
            x = self.contract_blocks[idx](x)
            features.append(x)
            x = self.pool_list[idx](x)
            
        return features
    
class ExpandPath(nn.Module):
    """This class consists of the """
    def __init__(self, chs=(1024,512,256,128,64)):
        super().__init__()
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i],chs[i+1],2,2) for i in range(len(chs)-1)])
        self.expand_blocks = nn.ModuleList(
            [DbleConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        
    def forward(self, x, encoder_features):
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            # TIP: Note that no cropping is necessary, since the feature maps
            # are the same dimensions spatially
            x = torch.cat([x, encoder_features[idx]], dim=1)
            x = self.expand_blocks[idx](x)
            
        return x
            
class UNet(nn.Module):
    def __init__(self, enc_chs=(1,64,128,256,512,1024),
                 dec_chs=(1024,512,256,128,64), n_class=1):
    
        super().__init__()
        self.encoder = ContractPath(enc_chs)
        self.decoder = ExpandPath(dec_chs)
        self.chan_combine = nn.Conv2d(dec_chs[-1], n_class, 1, padding=0)
        
    def forward(self, x):
        residual = x
        enc_features = self.encoder(x)
        # Reverse the order of the features list, use 0 as x, others as skips
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        out = self.chan_combine(out)
        
        out += residual

        return out
        
        
