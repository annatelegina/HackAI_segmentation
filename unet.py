
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UpConv, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            
        self.out_conv = nn.Conv2d((in_channels*3)//2, in_channels//2, kernel_size=3, padding=1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.out_conv(x)
        return x
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        self.help_conv = None
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels//2),
                    nn.ReLU(inplace=True),
                    #nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels//2, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    #nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    #nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    #nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))
    
class UNet_classic(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_classic, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.pre_out = nn.Conv2d(64, 3, 3, padding=1, bias=False)
        self.outc = OutConv(3, n_classes)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        inp = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.pre_out(x) # + inp
        x = self.outc(x)
        logits = self.out_act(x)

        return logits
    
class UNet_reduced(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_reduced, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.pre_out = nn.Conv2d(32, 3, 3, padding=1, bias=False)
        self.outc = OutConv(3, n_classes)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        inp = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.pre_out(x) 
        x = self.outc(x)
        logits = self.out_act(x)

        return logits
    
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y 

class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding = (kernel_size - 1)//2, bias=bias,))
            if bn: 
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: 
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

    
class UNet_RCAB(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_RCAB, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 32)
        self.pool = nn.MaxPool2d((2, 2))
        self.rcab1 = RCAB(32)
        self.conv1 = ConvReLU(32, 64)
        
        self.rcab2 = RCAB(64)
        self.conv2 = ConvReLU(64, 128)
        
        self.rcab3 = RCAB(128)
        self.conv3 = ConvReLU(128, 256)
        
        self.rcab4 = RCAB(256)
        self.conv4 = ConvReLU(256, 512)
        
        self.rcab5 = RCAB(512)
        #self.conv4 = ConvReLU(256, 256)
        
        self.conv_up1 = UpConv(512, 256)
        self.rcab_up1 = RCAB(256)

        self.conv_up2 = UpConv(256, 128)
        self.rcab_up2 = RCAB(128)

        self.conv_up3 = UpConv(128, 64)
        self.rcab_up3 = RCAB(64)
        
        self.conv_up4 = UpConv(64, 32)
        self.rcab_up4 = RCAB(32)

        self.outc = OutConv(32, n_classes)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        inp = x
        x0 = self.inc(x)
        x1 = self.conv1(self.rcab1(x0))
        pool_x1 = self.pool(x1)
        
        x2 = self.conv2(self.rcab2(pool_x1))
        pool_x2 = self.pool(x2)
        
        x3 = self.conv3(self.rcab3(pool_x2))
        pool_x3 = self.pool(x3)
        
        x4 = self.conv4(self.rcab4(pool_x3))
        pool_x4 = self.pool(x4)
        
        x5 = self.rcab5(pool_x4)

        up = self.conv_up1(x5, x4)
        x = self.rcab_up1(up)
        
        x = self.rcab_up2(self.conv_up2(x, x3))
        
        x = self.rcab_up3(self.conv_up3(x, x2))
        
        x = self.rcab_up4(self.conv_up4(x, x1))
        
        x = self.outc(x)
        logits = self.out_act(x)

        return logits
