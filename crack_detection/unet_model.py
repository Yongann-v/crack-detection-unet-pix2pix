"""
PyTorch UNet implementation for crack segmentation.
Save this as unet_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution + Batch Normalization + ReLU block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UNet(nn.Module):
    """
    UNet implementation for crack segmentation.
    
    Args:
        num_classes (int): Number of output classes
        align_corners (bool): Align corners for interpolation
        use_deconv (bool): Use deconvolution instead of bilinear upsampling
        in_channels (int): Number of input channels
        pretrained (str): Path to pretrained weights
    """
    
    def __init__(self,
                 num_classes,
                 align_corners=False,
                 use_deconv=False,
                 in_channels=3,
                 pretrained=None):
        super().__init__()

        self.encode = Encoder(in_channels)
        self.decode = Decoder(align_corners, use_deconv=use_deconv)
        self.cls = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        logit_list = []
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        """Initialize weights with Xavier normal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if self.pretrained is not None:
            self.load_pretrained_weights(self.pretrained)
    
    def load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights from file"""
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")


class Encoder(nn.Module):
    """UNet Encoder"""
    
    def __init__(self, in_channels=3):
        super().__init__()

        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3), 
            ConvBNReLU(64, 64, 3)
        )
        
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        self.down_sample_list = nn.ModuleList([
            self.down_sampling(channel[0], channel[1])
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(ConvBNReLU(in_channels, out_channels, 3))
        modules.append(ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class Decoder(nn.Module):
    """UNet Decoder"""
    
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.ModuleList([
            UpSampling(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Module):
    """UNet Upsampling Block"""
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False):
        super().__init__()

        self.align_corners = align_corners
        self.use_deconv = use_deconv
        
        if self.use_deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels,
                out_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                size=short_cut.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        x = torch.cat([x, short_cut], dim=1)
        x = self.double_conv(x)
        return x
