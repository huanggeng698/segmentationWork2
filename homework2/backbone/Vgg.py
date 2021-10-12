import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
from collections import OrderedDict

class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        self.pretrained_net = vgg16_bn(pretrained=pretrained)
        self.stage1 = self.pretrained_net.features[:7]
        self.stage2 = self.pretrained_net.features[7:14]
        self.stage3 = self.pretrained_net.features[14:24]
        self.stage4 = self.pretrained_net.features[24:34]
        self.stage5 = self.pretrained_net.features[34:]
        self.low_level_output = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(512, 2048, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(2048, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
    def forward(self, x):
        out = OrderedDict()
        low_level_features = self.stage2(self.stage1(x))
        x = self.stage5(self.stage4(self.stage3(low_level_features)))
        low_level_features = self.low_level_output(low_level_features)
        x = self.output_layer(x)
        out['out'] = x
        out['low_level'] = low_level_features
        return out


if __name__ == '__main__':
    image = torch.rand(1, 3, 512, 512)
    model = VGG(pretrained=True)
    model.eval()
    x, low = model(image)
    print('Done!')