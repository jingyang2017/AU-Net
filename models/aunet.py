import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .FAN import FAN_, ConvBlock
except:
    from FAN import FAN_, ConvBlock

class AUFAN_TDN_(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, n_classes=12):
        super(AUFAN_TDN_, self).__init__()
        self.alpha = alpha
        self.beta = beta
        num_input_channels = 256 * 3
        self.conv1x1 = nn.Conv2d(num_input_channels, 256, kernel_size=1, stride=1, padding=0)

        self.ConvNet_1 = []
        for in_f, out_f in [(256, 256)] * 1:
            self.ConvNet_1.append(ConvBlock(in_f, out_f))
            self.ConvNet_1.append(nn.MaxPool2d(2, 2))
        self.ConvNet_1 = nn.Sequential(*self.ConvNet_1)

        self.ConvNet_diff = []
        for in_f, out_f in [(256, 256)] * 1:
            self.ConvNet_diff.append(ConvBlock(in_f, out_f))
            self.ConvNet_diff.append(nn.MaxPool2d(2, 2))
        self.ConvNet_diff = nn.Sequential(*self.ConvNet_diff)

        self.ConvNet_2 = []
        for in_f, out_f in [(256, 256)] * 3:
            self.ConvNet_2.append(ConvBlock(in_f, out_f))
            self.ConvNet_2.append(nn.MaxPool2d(2, 2))
        self.ConvNet_2 = nn.Sequential(*self.ConvNet_2)

        self.predictor = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                       nn.Linear(128, n_classes))

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1x1_diff = nn.Conv2d(num_input_channels * 2, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x, x1, x2):
        x_diff = self.conv1x1_diff(self.avg_diff(torch.cat((x - x1, x2 - x), dim=1)))
        x = self.conv1x1(x)
        x = self.alpha * x + self.beta * F.interpolate(x_diff, x.size()[2:])
        x = self.ConvNet_1(x)
        x_diff = self.ConvNet_diff(x_diff)
        x = self.alpha * x + self.beta * F.interpolate(x_diff, x.size()[2:])
        x = self.ConvNet_2(x)
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), x.shape[1])
        predictions = self.predictor(x)
        return predictions


class AU_NET(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, n_classes=12):
        super(AU_NET, self).__init__()
        self.feature = FAN_()
        try:
            self.feature.fan.load_state_dict(torch.load('./models/2dfan2.pth', map_location='cpu'))
        except:
            self.feature.fan.load_state_dict(torch.load('2dfan2.pth', map_location='cpu'))
        self.feature.eval()
        for p in self.feature.parameters():
            p.requires_grad = False
        self.predictor = AUFAN_TDN_(alpha=alpha, beta=beta, n_classes=n_classes)

    def forward(self, x, x2, x3):
        with torch.no_grad():
            _, basic_feat1 = self.feature(x)
            _, basic_feat2 = self.feature(x2)
            _, basic_feat3 = self.feature(x3)
        predictions = self.predictor(basic_feat1, basic_feat2, basic_feat3)
        return predictions

if __name__ == '__main__':
    model = AU_NET()
    output = model(torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256))
    print(output.size())
    paras = sum([p.numel() for p in model.parameters()])
    print(paras)
