import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )




class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
        )

    def forward(self, x):
        return self.reduce(x)



class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=6):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(x)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)


class MAI(nn.Module):
    def __init__(self, channel):
        super(MAI, self).__init__()
        self.ms = Multi_scale(channel, channel)
        self.sal_res = ChannelAttention(channel)
        self.edg_res = ChannelAttention(channel)
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

        self.edg_conv = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.sal_conv = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0))
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, sal, edg):
        sal_res = self.sal_res(sal)
        edg_res = self.edg_res(sal)
        
        sal_r = self.ms(sal)
        sal_c = self.ca(sal_r) * sal_r
        sal_A = self.sa(sal_c) * sal_c
        edg_s = self.sigmoid(edg) * edg
        
        edg_o = self.edg_conv(edg_s * sal_A)
        sal_o = self.sal_conv(torch.cat((sal_A, edg_s), 1))

        return (sal_res + sal_o), (edg_res + edg_o)

class Multi_scale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Multi_scale, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        )
 

        self.conv_cat = BasicConv2d(5 * out_channel, out_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x + x0)
        x2 = self.branch2(x + x1)
        x3 = self.branch3(x + x2)
        x4 = self.branch4(x + x3)

        x_cat = torch.cat((x0, x1, x2, x3, x4), 1)
        x_cat = self.conv_cat(x_cat)
        return x_cat
        
    

class SF(nn.Module):
    def __init__(self, channel, eps=1e-8):
        super(SF, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.S_conv = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.weights1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, fl, fh, fs):
        fsl = F.interpolate(fs, size=fl.size()[2:], mode='bilinear')
        
        weights1 = nn.ReLU()(self.weights1)
        fuse_weights1 = weights1 / (torch.sum(weights1, dim=0) + self.eps)
        weights2 = nn.ReLU()(self.weights2)
        fuse_weights2 = weights1 / (torch.sum(weights2, dim=0) + self.eps)
        fs1 = fuse_weights1[0] * fsl 
        fl_1 = fuse_weights1[1] * fl 
        
        fs2 = fuse_weights2[0] * fsl 
        fh_2 = fuse_weights2[1] * fh 

        fl = self.conv2(fs1 * fl_1 + fl)
        fh = self.conv1(fs2 * fh_2 + fh)

        out = self.S_conv(torch.cat((fh, fl), 1))
        return out
    

class layer_fuse(nn.Module):
    def __init__(self, channel):
        super(layer_fuse, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, stride=2, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 1, padding=0)
        self.S_conv = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

    def forward(self, fl, fm, fh):
        fl = self.conv1(fl)
        # print(fl.size())
        fh = F.interpolate(fh, size=fm.size()[2:], mode='bilinear')
        # print(fh.size())
        fm = self.conv2(fm)
        # print(fm.size())
        out = self.S_conv(torch.cat((fl,fm,fh), 1))
        return out    

class GlobalFeature(nn.Module):
    """
    全局信息：由两部分组成 : 1.fc产生的全局信息；2.non-local方式的全局tezheng
    """
    def __init__(self, in_channels, channles, reduction=8):
        super(GlobalFeature, self).__init__()
        # ------------- fc 全局特征
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction, in_channels),
            nn.Sigmoid()
        )
        # ------------- non-block context
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(in_channels, channles, kernel_size=1),
            nn.LayerNorm([channles, 1, 1]),
            nn.Conv2d(channles, in_channels, kernel_size=1)
        )
        self.init_weight()
        

    def forward(self, x):
        batch, channel, height, weight = x.size()
        # -----------全局
        # [B, C*1*1]
        avg_pool = self.avg_pool(x).view(batch, channel)
        avg_feature = self.fc1(avg_pool).view(batch, channel, 1, 1)
        out1 = torch.mul(x, avg_feature)
        # -----------non
        input_x = x
        # [B, C, H*W]
        input_x = input_x.view(batch, channel, height*weight)
        # [B, 1, C, H*W]
        input_x = input_x.unsqueeze(1)
        # [B, 1, H, W]
        conv_mask = self.conv_mask(x)
        # [B, 1, H*W]
        conv_mask = conv_mask.view(batch, 1, height*weight)
        # [B, 1, H*w]
        context_mask = self.softmax(conv_mask)
        # [B, 1, H*W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [B, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [B, C, 1, 1]
        context = context.view(batch, channel, 1, 1)


        channel_mul = self.channel_mul_conv(context)
        out2 = torch.mul(x, channel_mul)
        return out1 + out2
    
    
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.reduce_sal1 = Reduction(encoder_channels[-4], decode_channels)
        self.reduce_sal2 = Reduction(encoder_channels[-3], decode_channels)
        self.reduce_sal3 = Reduction(encoder_channels[-2], decode_channels)
        self.reduce_sal4 = Reduction(encoder_channels[-1], decode_channels)

        self.reduce_edg1 = Reduction(encoder_channels[-4], decode_channels)
        self.reduce_edg2 = Reduction(encoder_channels[-3], decode_channels)
        self.reduce_edg3 = Reduction(encoder_channels[-2], decode_channels)
        self.reduce_edg4 = Reduction(encoder_channels[-1], decode_channels)

        #边缘、语义信息输出流
        self.S1 = nn.Sequential(
            BasicConv2d(decode_channels, decode_channels, 3, padding=1),
            nn.Conv2d(decode_channels, decode_channels, 1)
        )
        self.S2 = nn.Sequential(
            BasicConv2d(decode_channels, decode_channels, 3, padding=1),
            nn.Conv2d(decode_channels, 1, 1)
        )
        self.S3 = nn.Sequential(
            BasicConv2d(decode_channels, decode_channels, 3, padding=1),
            nn.Conv2d(decode_channels, decode_channels, 1)
        )
        self.S4 = nn.Sequential(
            BasicConv2d(decode_channels, decode_channels, 3, padding=1),
            nn.Conv2d(decode_channels, 1, 1)
        )
        self.S5 = nn.Sequential(
            BasicConv2d(decode_channels, decode_channels, 3, padding=1),
            nn.Conv2d(decode_channels, decode_channels, 1)
        )
        self.S6 = nn.Sequential(
            BasicConv2d(decode_channels, decode_channels, 3, padding=1),
            nn.Conv2d(decode_channels, 1, 1)
        )
        self.S7 = nn.Sequential(
            BasicConv2d(decode_channels, decode_channels, 3, padding=1),
            nn.Conv2d(decode_channels, decode_channels, 1)
        )
        self.S8 = nn.Sequential(
            BasicConv2d(decode_channels, decode_channels, 3, padding=1),
            nn.Conv2d(decode_channels, 1, 1)
        )
        # 边缘信息流
        self.S_conv1 = nn.Sequential(
            BasicConv2d(2 * decode_channels, decode_channels, 3, padding=1),
            BasicConv2d(decode_channels, decode_channels, 1)
        )
        self.S_conv2 = nn.Sequential(
            BasicConv2d(2 * decode_channels, decode_channels, 3, padding=1),
            BasicConv2d(decode_channels, decode_channels, 1)
        )
        self.S_conv3 = nn.Sequential(
            BasicConv2d(2 * decode_channels, decode_channels, 3, padding=1),
            BasicConv2d(decode_channels, decode_channels, 1)
        )
        self.S_conv4 = nn.Sequential(
            BasicConv2d(2 * decode_channels, decode_channels, 3, padding=1),
            BasicConv2d(decode_channels, decode_channels, 1)
        )

        self.global_info1 = GlobalFeature(decode_channels, decode_channels)
        self.global_info2 = GlobalFeature(decode_channels, decode_channels)
        self.global_info3 = GlobalFeature(decode_channels, decode_channels)
        self.global_info4 = GlobalFeature(decode_channels, decode_channels)

        
        self.sigmoid = nn.Sigmoid()

        self.mai4 = MAI(decode_channels)
        self.mai3 = MAI(decode_channels)
        self.mai2 = MAI(decode_channels)
        self.mai1 = MAI(decode_channels)

        self.sf1 = SF(decode_channels)
        self.sf2 = SF(decode_channels)


        if self.training:
            self.aux_head = AuxHead(decode_channels, num_classes)


        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        #语义特征分支
        x_sal1 = self.reduce_sal1(res1)      
        x_sal2 = self.reduce_sal2(res2)
        x_sal3 = self.reduce_sal3(res3)
        x_sal4 = self.reduce_sal4(res4)
        
        #全局注意力模块
        x_sal1 = self.global_info1(x_sal1)
        x_sal2 = self.global_info2(x_sal2)
        x_sal3 = self.global_info3(x_sal3)
        x_sal4 = self.global_info4(x_sal4)
        
        #边缘特征分支
        x_edg1 = self.reduce_edg1(res1)
        x_edg2 = self.reduce_edg2(res2)
        x_edg3 = self.reduce_edg3(res3)
        x_edg4 = self.reduce_edg4(res4)
        
        #FMI-4的输出
        sal4, edg4 = self.mai4(x_sal4, x_edg4)
        #上采样2倍
        sal4_3 = F.interpolate(sal4, size=x_sal3.size()[2:], mode='bilinear')
        edg4_3 = F.interpolate(edg4, size=x_sal3.size()[2:], mode='bilinear')
        
        #Cat信息融合
        x_sal3 = self.S_conv1(torch.cat((sal4_3, x_sal3),1))
        x_edg3 = self.S_conv2(torch.cat((edg4_3, x_edg3),1))
        
        #FMI-3的输出
        sal3, edg3 = self.mai3(x_sal3, x_edg3)
        #上采样2倍
        sal3_2 = F.interpolate(sal3, size=x_sal2.size()[2:], mode='bilinear')
        edg3_2 = F.interpolate(edg3, size=x_sal2.size()[2:], mode='bilinear')
        
        #语义SF信息融合，边缘Cat信息融合
        x_sal2 = self.sf1(x_sal2, sal3_2, sal4)
        x_edg2 = self.S_conv3(torch.cat((edg3_2, x_edg2),1))
        
        #FMI-2的输出
        sal2, edg2 = self.mai2(x_sal2, x_edg2)
        sal2_1 = F.interpolate(sal2, size=x_sal1.size()[2:], mode='bilinear')
        edg2_1 = F.interpolate(edg2, size=x_sal1.size()[2:], mode='bilinear')
        #语义SF信息融合，边缘Cat信息融合
        x_sal1 = self.sf2(x_sal1, sal2_1, sal4)
        x_edg1 = self.S_conv4(torch.cat((edg2_1, x_edg1), 1))
        #FMI-1的输出
        sal1, edg1 = self.mai1(x_sal1, x_edg1)
        
        #各尺度边缘，语义特征用于监督训练，计算损失
        sal_out = self.S1(sal1)

        #语义特征送入分割头，并上采样至原图大小
        x = self.segmentation_head(sal_out)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:
            edg_out = self.S2(edg1)
            sal2 = self.S3(sal2)
            edg2 = self.S4(edg2)
            sal3 = self.S5(sal3)
            edg3 = self.S6(edg3)
            sal4 = self.S7(sal4)
            edg4 = self.S8(edg4)
            edg_out = F.interpolate(edg_out, size=(h,w), mode='bilinear', align_corners=True)
            sal2 = self.aux_head(sal2,h,w)
            edg2 = F.interpolate(edg2, size=(h,w), mode='bilinear', align_corners=True)
            sal3 = self.aux_head(sal3,h,w)
            edg3 = F.interpolate(edg3, size=(h,w), mode='bilinear', align_corners=True)
            sal4 = self.aux_head(sal4,h,w)
            edg4 = F.interpolate(edg4, size=(h,w), mode='bilinear', align_corners=True)
            return x, edg_out, sal2, edg2,  sal3,  edg3, sal4,  edg4
        else:
            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


from timm.models.efficientnet import _cfg

class MFIN(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()
        config = _cfg(url='', file='pretrain_weights/semi_weakly_supervised_resnet18.pth')
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                        out_indices=(1, 2, 3, 4), pretrained=pretrained,pretrained_cfg=config)
        
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        # print(h, w)
        res1, res2, res3, res4 = self.backbone(x)
        # print(res1.size(),res2.size(),res3.size(),res4.size())
        #torch.Size([8, 64, 256, 256]) torch.Size([8, 128, 128, 128]) #1/4;  1/8
        #torch.Size([8, 256, 64, 64]) torch.Size([8, 512, 32, 32])  #1/16;  1/32
        if self.training:
            x,  edg1, s2,  edg2, s3,  edg3, s4, edg4 = self.decoder(res1, res2, res3, res4, h, w)
            # print("output:",x.size())
            # print("edg1:", edg1.size())
            return x,  edg1, s2,  edg2, s3, edg3, s4,  edg4
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x


from thop import profile
if __name__ == "__main__":
    model = MFIN(num_classes=6)
    input = torch.randn(1, 3, 512, 512)
    Flops, params = profile(model, inputs=(input,)) # macs
    print('Flops: % .4fG'%(Flops / 1000000000))# 计算量
    print('params参数量: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值
    
#512x512    
# Flops:  34.2356G
# params参数量:  15.3776M
