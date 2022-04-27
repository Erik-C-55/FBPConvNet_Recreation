import torch
import torch.nn as nn

# NOTES: This implements the modified U-Net architecture described in [1]. The
#       code is based heavily off on [2], with slight modifications made to 
#       match the feature map sizes in [1] and facilitate model viewing in 
#       Tensorboard. The Tensorboard modifications make the code less Pythonic,
#       but clean up the model graph in Tensorboard. Since this code implements
#       a very specific version of U-Net, this loss of flexibility is okay.

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
        
        self.stage1 = DbleConvBlock(chs[0],chs[1])
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = DbleConvBlock(chs[1],chs[2])
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = DbleConvBlock(chs[2],chs[3])
        self.pool3 = nn.MaxPool2d(2)
        self.stage4 = DbleConvBlock(chs[3],chs[4])
        self.pool4 = nn.MaxPool2d(2)
        self.stage5 = DbleConvBlock(chs[4],chs[5])
        
    def forward(self, x):
        # Along the way, save features for expansion path
        features =[]
        
        x = self.stage1(x)
        features.append(x)
        x = self.pool1(x)
        
        x = self.stage2(x)
        features.append(x)
        x = self.pool2(x)
        
        x = self.stage3(x)
        features.append(x)
        x = self.pool3(x)
        
        x = self.stage4(x)
        features.append(x)
        x = self.pool4(x)
        
        x = self.stage5(x)
        features.append(x)
            
        return features
    
class ExpandPath(nn.Module):
    """This class consists of the """
    def __init__(self, chs=(1024,512,256,128,64)):
        super().__init__()
        
        self.upconv1 = nn.ConvTranspose2d(chs[0],chs[1],2,2)
        self.stage1 = DbleConvBlock(chs[0],chs[1])
        self.upconv2 = nn.ConvTranspose2d(chs[1],chs[2],2,2)
        self.stage2 = DbleConvBlock(chs[1],chs[2])
        self.upconv3 = nn.ConvTranspose2d(chs[2],chs[3],2,2)
        self.stage3 = DbleConvBlock(chs[2],chs[3])
        self.upconv4 = nn.ConvTranspose2d(chs[3],chs[4],2,2)
        self.stage4 = DbleConvBlock(chs[3],chs[4])
        
    def forward(self, x, encoder_features):
        # Along the way, concatenate features from contracting path
        x = self.upconv1(x)
        x = torch.cat([x, encoder_features[0]], dim=1)
        x = self.stage1(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, encoder_features[1]], dim=1)
        x = self.stage2(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, encoder_features[2]], dim=1)
        x = self.stage3(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, encoder_features[3]], dim=1)
        x = self.stage4(x)
            
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
        out = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        out = self.chan_combine(out)
        
        out += residual

        return out
        
        
