import torch
import torch.nn as nn

try:
    from .FAN import FAN_, ConvBlock
except:
    from FAN import FAN_, ConvBlock

'''
code is modified from https://github.com/face-analysis/emonet/blob/master/emonet/models/emonet.py
'''
class AUFAN_(nn.Module):
    def __init__(self, n_classes=12, dim=256):
        super(AUFAN_, self).__init__()
        num_input_channels = 256*3
        n_blocks = 4
        self.conv1x1 = nn.Conv2d(num_input_channels, dim, kernel_size=1, stride=1, padding=0)
        self.ConvNet = []
        for in_f, out_f in [(dim, dim)] * n_blocks:
            self.ConvNet.append(ConvBlock(in_f, out_f))
            self.ConvNet.append(nn.MaxPool2d(2, 2))
        self.ConvNet = nn.Sequential(*self.ConvNet)
        self.avg_pool = nn.AvgPool2d(4)
        self.predictor = nn.Sequential(nn.Linear(dim, dim//2), nn.BatchNorm1d(dim//2), nn.ReLU(inplace=True),nn.Linear(dim//2, n_classes))

    def forward(self, x):
        feats = self.conv1x1(x)
        feats = self.ConvNet(feats)
        feat = self.avg_pool(feats)
        feat = feat.view(feat.size(0), feat.size(1))
        predictions = self.predictor(feat)
        return predictions

class AU_FAN(nn.Module):
    def __init__(self, n_classes=12, dim=256):
        super(AU_FAN, self).__init__()
        self.feature = FAN_()
        try:
            self.feature.fan.load_state_dict(torch.load('./models/2dfan2.pth',map_location='cpu'))
        except:
            self.feature.fan.load_state_dict(torch.load('2dfan2.pth',map_location='cpu'))

        self.feature.eval()
        for p in self.feature.parameters():
            p.requires_grad = False

        self.predictor = AUFAN_(n_classes=n_classes, dim=dim)

    def forward(self, x):
        with torch.no_grad():
            hms, basic_feat = self.feature(x)
        predictions = self.predictor(basic_feat)
        if self.training:
            return hms, predictions
        else:
            return predictions
åå
if __name__ == '__main__':
    model = AUFAN()
    model.eval()
    b = model(torch.randn(1,3,256,256))


