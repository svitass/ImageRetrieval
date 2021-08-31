import torch.nn as nn
import torch

# 构造50层的残差块 conv2.x---conv5.x
class Bottleneck(nn.Module):

    def __init__(self,in_channel,out_chanels,stride=1,isDownsample=True):
        super(Bottleneck,self).__init__()
        out1,out2,out3 = out_chanels
        # 1*1,3*3,1*1 第一个1*1只是为了改变输出通道数 3*3的卷积可能改变卷积核大小，要计算stride
        self.conv1 = nn.Conv2d(in_channel,out1,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out1)

        self.conv2 = nn.Conv2d(out1,out2,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out2)

        self.conv3 = nn.Conv2d(out2,out3,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out3)
        self.relu = nn.ReLU(inplace=True)
        if isDownsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel,out3,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out3)
            )
        else:
            self.downsample = None


    def forward(self,x):
        residual = x
        # 保证残差维度和输出一样
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 先加上残差，再激活函数
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(ResNet,self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64), # 进行数据的归一化处理
            nn.ReLU(True),
            nn.MaxPool2d(3,2,padding=1),  # pool操作不改变张量的通道数
        )
        self.stage2 = self._make_layer(64,3,[64, 64, 256], stride=1)
        self.stage3 = self._make_layer(256,4,[128,128,512],stride=2)
        self.stage4 = self._make_layer(512,6,[256,256,1024],stride=2)
        self.stage5 = self._make_layer(1024,3,[512,512,2048],stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1,1))  # 指定输出尺寸为(1,1) 自适应平均池化操作
        self.fc = nn.Sequential(
            nn.Linear(2048,num_classes)
        )

    def _make_layer(self,in_channel,block_num,channels,stride=1):
        # 第一个block,需要dowmsample
        layers = [
            Bottleneck(in_channel,channels,stride=stride,isDownsample=True)
        ]
        in_channel = channels[2]  # 最后一个输出通道作为输入通道
        # 剩下block_num -1个block,不需要下采样
        for i in range(1,block_num):
            layers.append(
                Bottleneck(in_channel,channels,isDownsample=False)
            )
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.pool(out)
        out = torch.flatten(out,1)  # 展平处理 返回只有一维的数据
        out = self.fc(out)  # fc的维度等于N*C*H*W
        return  out
