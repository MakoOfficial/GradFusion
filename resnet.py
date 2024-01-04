import torch
import torchvision
from torch import nn
from torch.nn import functional as F
# from torchsummary import summary


class ResidualBlock(nn.Module):
    """
    实现子module: Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    """
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    """

    def __init__(self, blocks, start_channel=8, num_classes=230):
        super(ResNet, self).__init__()
        self.model_name = 'resnet18'

        # 前几层: 图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(start_channel, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 64, blocks[0])
        self.layer2 = self._make_layer(64, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, blocks[3], stride=2)

        # 分类用的全连接
        self.MLP = nn.Linear(512+32, num_classes)
        self.softmax = nn.Softmax()

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer,包含多个residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x, gender):
        x = self.pre(x)
        l1_out = self.layer1(x)
        l2_out = self.layer2(l1_out)
        l3_out = self.layer3(l2_out)
        l4_out = self.layer4(l3_out)
        p_out = F.adaptive_avg_pool2d(l4_out, 1)
        fea = p_out.view(p_out.size(0), -1)

        gender_encode = self.gender_encoder(gender)

        out = self.MLP(torch.cat((fea, gender_encode), dim=-1))
        return l1_out, l2_out, l3_out, l4_out, fea, self.softmax(out)

def ResNet18():
    return ResNet([2, 2, 2, 2])


def ResNet34():
    return ResNet([3, 4, 6, 3])


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18()
    model.to(device)
    # summary(model, (3, 224, 224))
    print(f'fusion:{sum(p.nelement() for p in model.parameters() if p.requires_grad == True) / 1e6}M')