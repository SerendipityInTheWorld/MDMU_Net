import torch
import torch.nn as nn
import numpy as np

class Conv_blockbone(nn.Module):
    def __init__(self,in_channels,drop=0.3):
        super(Conv_blockbone,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv1(x)
        result = self.conv2(x)
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

        self.encode_1 = Conv_blockbone(32,drop=0.3)
        self.dowsample_1 = DownsamplingConv(32,64)

        self.encode_2 = Conv_blockbone(64,drop=0.3)
        self.dowsample_2 = DownsamplingConv(64, 128)

        self.encode_3 = Conv_blockbone(128,drop=0.3)
        self.dowsample_3 = DownsamplingConv(128, 256)

        self.encode_4 = Conv_blockbone(256,drop=0.3)
        self.dowsample_4 = DownsamplingConv(256, 512)

        self.bottom_conv = Bottom_conv(512)


    def forward(self,x):
        # stage1
        x0 = self.conv_trans(x)
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

class UNetRes(nn.Module):
    def __init__(self,class_num=3):
        super(UNetRes,self).__init__()
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
    model = UNetRes(class_num=3)
    random_3d_tensor = torch.rand(1, 1, 96, 96,96 )
    print(random_3d_tensor.shape)
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
    # FLOPs
    flops, _ = profile(model, inputs=(input,))
    # FLOPs
    gflops = flops / 1e9
    print(f"FLOPs: {gflops:.3f}G")