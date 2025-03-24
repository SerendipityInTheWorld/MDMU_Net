import torch
import torch.nn as nn
import numpy as np
from lib.LN import LayerNorm

class PointwiseConv(nn.Module):
    def __init__(self,in_channels,dimg_Expand=4,layer_scale_init_value=1e-6):
        super(PointwiseConv,self).__init__()
        self.pwconv1 = nn.Conv3d(in_channels, dimg_Expand * in_channels, kernel_size=1, groups=in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(dimg_Expand * in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(1,in_channels,1,1,1),
                                  requires_grad=True)
    def forward(self,x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * self.gamma.to(x.device)
        return x
class DepthGating(nn.Module):
    def __init__(self,in_channels,kernel_size=5):
        super(DepthGating,self).__init__()
        self.dwc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.Conv3d(in_channels, in_channels,
                      kernel_size=kernel_size,
                      padding=(kernel_size-1)//2,
                      groups=in_channels)
        )
        self.conv_1x1x1 = nn.Conv3d(in_channels,in_channels,kernel_size=1)
        self.sigmod = nn.Sigmoid()

    def forward(self,x):
        x1 = self.conv_1x1x1(x)
        y_weight = self.sigmod(self.dwc(x))

        result = x1 * y_weight
        return result

class MDDMS(nn.Module):
    """
    Depthwise Pointwise Convolution
    """
    def __init__(self,in_channels,
                 arxis=None,):
        super(MDDMS,self).__init__()
        self.kernel_size = []
        self.padding = []
        if arxis == 'D':
            self.kernel_size.append((5, 3, 3))
            self.kernel_size.append((7, 3, 3))
            self.kernel_size.append((9, 3, 3))
            self.padding.append((2, 1, 1))
            self.padding.append((3, 1, 1))
            self.padding.append((4, 1, 1))
        elif arxis == 'H':
            self.kernel_size.append((3, 5, 3))
            self.kernel_size.append((3, 7, 3))
            self.kernel_size.append((3, 9, 3))
            self.padding.append((1, 2, 1))
            self.padding.append((1, 3, 1))
            self.padding.append((1, 4, 1))
        elif arxis == 'W':
            self.kernel_size.append((3, 3, 5))
            self.kernel_size.append((3, 3, 7))
            self.kernel_size.append((3, 3, 9))
            self.padding.append((1, 1, 2))
            self.padding.append((1, 1, 3))
            self.padding.append((1, 1, 4))

        self.depthconv_1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=self.kernel_size[0],
                      padding=self.padding[0],
                      groups=in_channels),
            PointwiseConv(in_channels)
        )

        self.depthconv_2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=self.kernel_size[1],
                      padding=self.padding[1],
                      groups=in_channels),
            PointwiseConv(in_channels)
        )

        self.depthconv_3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=self.kernel_size[2],
                      padding=self.padding[2],
                      groups=in_channels),
            PointwiseConv(in_channels)
        )

        self.multiscale_fussion = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )

        self.DG1 = DepthGating(in_channels)
        self.DG2 = DepthGating(in_channels)
        self.DG3 = DepthGating(in_channels)

    def get_MDDMS(self,x):
        """
        LargeKernelMultiScale
        :return:
        """

        out1 = self.DG1(self.depthconv_1(x))
        out2 = self.DG2(self.depthconv_2(x))
        out3 = self.DG3(self.depthconv_3(x))
        result = out1 * out2 * out3 + x
        lkm_result = self.multiscale_fussion(result)
        return lkm_result

    def forward(self,x):
        result = self.get_MDDMS(x)
        return result
class MDMSE(nn.Module):
    def __init__(self,in_channels,drop=0.1):
        super(MDMSE,self).__init__()
        self.norm_ln1 = LayerNorm(in_channels)
        self.norm_ln2 = LayerNorm(in_channels)

        self.mpfe_D = MDDMS(in_channels,arxis='D')
        self.mpfe_H = MDDMS(in_channels,arxis='H')
        self.mpfe_W = MDDMS(in_channels,arxis='W')
        self.dwp = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.drop = nn.Dropout3d(p=drop)

        self.conv_1x1x1_dhw = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(inplace=True)
        )

        # 多维融合卷积
        self.fusion_conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels)
        )

        # 多维信息与res细节融合
        self.fusion_conv2 = nn.Sequential(
            nn.Conv3d(2*in_channels, 2*in_channels, kernel_size=3,padding=1,groups=2*in_channels),
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(2*in_channels,in_channels,kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(inplace=True)
        )

    def MultidimensionalExtraction(self,x):
        """多维度多尺度特征提取"""

        out_d = self.conv_1x1x1_dhw(self.mpfe_D(x))
        out_h = self.conv_1x1x1_dhw(self.mpfe_H(x))
        out_w = self.conv_1x1x1_dhw(self.mpfe_W(x))
        out_ = self.dwp(x) + x

        # out_xyz = self.fusion_conv1x1(torch.cat((out_d, out_h, out_w), dim=1))
        out_dhw = self.fusion_conv1x1(out_d * out_h * out_w)
        out = self.fusion_conv2(torch.cat([out_,out_dhw],dim=1))
        result = self.drop(out)

        return result
    def forward(self,x):
        # 前半部分
        # LN归一化
        x = self.norm_ln1(x)
        ME_result = self.MultidimensionalExtraction(x)
        x1 = ME_result + x

        # 后半部分
        x2 = self.norm_ln2(x1)
        result = self.conv_block(x2)
        return result

class ResConvBlockbone(nn.Module):
    def __init__(self,in_channels,out_channels,structure='decode',drop=0.1):
        super(ResConvBlockbone,self).__init__()
        self.sturcture = structure

        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.InstanceNorm3d(out_channels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        if self.sturcture == 'decode':
            self.drop = nn.Dropout3d(p=drop)

    def forward(self,x):
        source_x = x
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        if self.sturcture == 'decode':
            out2 = self.drop(out2)
        result = source_x + out2


        return result

class DownsamplingConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingConv,self).__init__()
        """
        先进行归一化再下采样，最后激活
        """
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        result = self.downsample(x)
        return result

class UpsamplingConv(nn.Module):
    def __init__(self, in_channels, out_channels,last=False):
        super(UpsamplingConv,self).__init__()

        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.skip_res = ResConvBlockbone(out_channels,out_channels,drop=0.05)
        if last:
            self.cat_conv = nn.Conv3d(2 * out_channels, out_channels, kernel_size=1)
        else:
            self.cat_conv = nn.Sequential(
                nn.Conv3d(2 * out_channels, out_channels, kernel_size=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, inp, skip):
        inp = self.transpose_conv(inp)
        skip = self.skip_res(skip)
        out = self.cat_conv(torch.cat([inp, skip], dim=1))
        return out

class Bottom_conv(nn.Module):
    def __init__(self, channel):
        super(Bottom_conv, self).__init__()
        self.res_conv = nn.Sequential(
            nn.Conv3d(channel,channel,kernel_size=3,padding=1,groups=channel),
            nn.Conv3d(channel, channel, kernel_size=1),
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel),
            nn.Conv3d(channel,channel,kernel_size=1),
        )
    def forward(self, x):
        source_x = x
        x_out = self.res_conv(x)
        result = x_out + source_x
        return result

class Seghead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Seghead, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        result = self.conv(x)
        return result

class Encode_net(nn.Module):
    def __init__(self):
        super(Encode_net,self).__init__()
        self.conv_trans = nn.Conv3d(1,32,kernel_size=7,stride=2,padding=3)
        # self.ln = LayerNorm(32)

        self.encode_1 = MDMSE(32,drop=0.3)
        self.dowsample_1 = DownsamplingConv(32,64)

        self.encode_2 = MDMSE(64,drop=0.3)
        self.dowsample_2 = DownsamplingConv(64, 128)

        self.encode_3 = MDMSE(128,drop=0.3)
        self.dowsample_3 = DownsamplingConv(128, 256)

        self.encode_4 = MDMSE(256,drop=0.3)
        self.dowsample_4 = DownsamplingConv(256, 512)

        self.bottom_conv = Bottom_conv(512)


    def forward(self,x):
        # stage1
        x0 = self.conv_trans(x)
        # x0 = self.ln(x0)
        x1 = self.encode_1(x0)
        # stage2
        x1_1 = self.dowsample_1(x1)
        x2 = self.encode_2(x1_1)
        # stage3
        x2_2 = self.dowsample_2(x2)
        x3 = self.encode_3(x2_2)
        # stage4
        x3_3 = self.dowsample_3(x3)
        x4 = self.encode_4(x3_3)
        # 底部
        x4_4 = self.dowsample_4(x4)
        x5 = self.bottom_conv(x4_4)
        return x1,x2,x3,x4,x5
class Decode_net(nn.Module):
    def __init__(self):
        super(Decode_net,self).__init__()

        self.up_1 = UpsamplingConv(512,256)
        self.decode_1 = ResConvBlockbone(256,256,structure='decode',drop=0.2)

        self.up_2 = UpsamplingConv(256, 128)
        self.decode_2 = ResConvBlockbone(128, 128,structure='decode',drop=0.2)

        self.up_3 = UpsamplingConv(128, 64)
        self.decode_3 = ResConvBlockbone(64, 64,structure='decode',drop=0.2)

        self.up_4 = UpsamplingConv(64, 32)
        self.decode_4 = ResConvBlockbone(32,32,structure='decode',drop=0.2)

        self.up = UpsamplingConv(32, 16,last=True)

    def forward(self,source_x,x1,x2,x3,x4,x5):
        # print('-----------------------------')
        # print('decode:')
        y = self.decode_1(self.up_1(x5,x4))
        # print(y.shape)
        y = self.decode_2(self.up_2(y,x3))
        # print(y.shape)
        y = self.decode_3(self.up_3(y,x2))
        # print(y.shape)
        y = self.decode_4(self.up_4(y,x1))
        # print(y.shape)
        y = self.up(y,source_x)

        return y

class UNetMDMSE(nn.Module):
    def __init__(self,class_num=3):
        super(UNetMDMSE,self).__init__()
        self.encode = Encode_net()
        self.decode = Decode_net()

        self.conv = nn.Conv3d(1,16,kernel_size=3,padding=1)
        self.outconv = Seghead(16,class_num)

    def forward(self,x):
        source_x = self.conv(x)

        x1,x2,x3,x4,x5 = self.encode(x)
        out = self.decode(source_x,x1,x2,x3,x4,x5)

        result = self.outconv(out)
        return result


if __name__ == '__main__':
    model = UNetMDMSE(class_num=3)
    # 生成随机3D Tensor，值范围默认为[0, 1)
    random_3d_tensor = torch.rand(1, 1, 96, 96,96 )
    print(random_3d_tensor.shape)
    print('====================')
    result = model(random_3d_tensor)
    print(result.shape)
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    print(f"Total params: {Total_params / 1000000}")
    print(f"Trainable params: {Trainable_params}")
    print(f"Non-trainable params: {NonTrainable_params}")


    from thop import profile, clever_format
    input = torch.randn(1, 1, 96, 96, 96)
    # 计算 FLOPs
    flops, _ = profile(model, inputs=(input,))
    # GFLOPs
    gflops = flops / 1e9
    print(f"FLOPs: {gflops:.3f}G")