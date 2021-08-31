import torch.nn as nn
from torchvision import models
import torch
import os

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese , self).__init__()
        self.model_user = models.resnet50(pretrained=False)   # user domain
        # cwd = os.getcwd()
        model_weight_path = "./resnet50_pre.pth"
        self.model_user.load_state_dict(torch.load(model_weight_path))
        self.model_shop = models.resnet50(pretrained=False)   # shop domain
        self.model_shop.load_state_dict(torch.load(model_weight_path))
        self.maxpool = nn.AdaptiveMaxPool2d((1,1)) # c=2048,h*w=7*7
        self.fc_user = nn.Linear(2048,2048)
        self.fc_shop = nn.Linear(2048,2048)
        # L2 正则化
        self.l2 = nn.functional.normalize

    def forward_user(self,q):  # query image 提取特征
        out = self.model_user.conv1(q)
        out = self.model_user.bn1(out)
        out = self.model_user.relu(out)
        out = self.model_user.maxpool(out)
        out = self.model_user.layer1(out)
        out = self.model_user.layer2(out)
        out = self.model_user.layer3(out)
        out = self.model_user.layer4(out)
        out = self.maxpool(out)
        out = torch.flatten(out , 1)  # 展平处理 返回只有一维的数据
        out = self.fc_user(out)
        out = self.l2(out)
        return out

    def forward_shop(self,x):  # relevant image,non-relevant-image提取特征
        x = self.model_shop.conv1(x)
        x = self.model_shop.bn1(x)
        x = self.model_shop.relu(x)
        x = self.model_shop.maxpool(x)
        x = self.model_shop.layer1(x)
        x = self.model_shop.layer2(x)
        x = self.model_shop.layer3(x)
        x = self.model_shop.layer4(x)
        x = self.maxpool(x)
        x = torch.flatten(x , 1)  # 展平处理 返回只有一维的数据
        x = self.fc_shop(x)
        x = self.l2(x)
        return x

    def forward(self,q,p,n): # query image,relevant image,non-relevant image
        q = self.forward_user(q)
        p = self.forward_shop(p)
        n = self.forward_shop(n)
        return [q,p,n]