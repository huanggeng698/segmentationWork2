from deeplabv3Plus import deeplabv3plus_vgg
from deeplabv3Plus import deeplabv3plus_xception
from deeplabv3Plus import deeplabv3plus_resnet50
from deeplabv3Plus import deeplabv3plus_resnet101
import torch


if __name__ == '__main__':
    image = torch.rand(1, 3, 513, 513)
    xception = deeplabv3plus_xception(num_classes=21, output_stride=16, pretrained_backbone=False)
    vgg = deeplabv3plus_vgg(num_classes=21, output_stride=16, pretrained_backbone=True)
    resnet50 = deeplabv3plus_resnet50(num_classes=21, output_stride=16, pretrained_backbone=True)
    resnet101 = deeplabv3plus_resnet101(num_classes=21, output_stride=16, pretrained_backbone=True)
    print('The input size is {}'.format(image.size()))
    print('Start testing......')
    print('Test xception......')
    xception.eval()
    out = xception(image)
    print('Output size is {}'.format(out.size()))

    print('Test vgg......')
    vgg.eval()
    out = vgg(image)
    print('Output size is {}'.format(out.size()))

    print('Test resnet50......')
    resnet50.eval()
    out = xception(image)
    print('Output size is {}'.format(out.size()))

    print('Test resnet101......')
    resnet101.eval()
    out = resnet101(image)
    print('Output size is {}'.format(out.size()))



