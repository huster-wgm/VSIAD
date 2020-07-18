import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = [
    'berg', 'bergVGG'
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
}


class Berg(nn.Module):
    """
    Refer ==> Berg, Amanda, Jorgen Ahlberg, and Michael Felsberg. \
    "Generating visible spectrum images from thermal infrared." \
    CVPR Workshops. 2018.
    """
    def __init__(self, src_ch, tar_ch):
        super(Berg, self).__init__()
        kernels = [32, 64, 128, 128]
        self.downConv1 = nn.Sequential(
            nn.Conv2d(src_ch, kernels[0], 3, padding=1),
            # nn.BatchNorm2d(kernels[0]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.downConv2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], 3, padding=1),
            # nn.BatchNorm2d(kernels[1]),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),)

        self.downConv3 = nn.Sequential(
            nn.Conv2d(kernels[1], kernels[2], 3, padding=1),
            # nn.BatchNorm2d(kernels[2]),
            nn.LeakyReLU(inplace=True),)

        self.downConv4 = nn.Sequential(
            nn.Conv2d(kernels[2], kernels[3], 3, padding=1),
            # nn.BatchNorm2d(kernels[3]),
            nn.LeakyReLU(inplace=True),)

        self.upConv1 = nn.Sequential(
            nn.Conv2d(kernels[3], kernels[3], 3, padding=1),
            # nn.BatchNorm2d(kernels[3]),
            # nn.Dropout2d(p=0.5, inplace=True),
            nn.ReLU(inplace=True),)

        self.upConv2 = nn.Sequential(
            nn.Conv2d(kernels[3]+kernels[2], kernels[2], 3, padding=1),
            # nn.BatchNorm2d(kernels[2]),
            # nn.Dropout2d(p=0.5, inplace=True),
            nn.ReLU(inplace=True),)

        self.upConv3 = nn.Sequential(
            nn.Conv2d(kernels[2]+kernels[1], kernels[1], 3, padding=1),
            # nn.BatchNorm2d(kernels[1]),
            # nn.Dropout2d(p=0.5, inplace=True),
            nn.ReLU(inplace=True),)

        # blue in paper
        self.upConv4 = nn.Sequential(
            nn.Conv2d(kernels[1]+kernels[0], kernels[0], 3, padding=1),
            # nn.BatchNorm2d(kernels[0]),
            # nn.Dropout2d(p=0.5, inplace=True),
            nn.ReLU(inplace=True),)

        self.convPred = nn.Sequential(
            nn.Conv2d(kernels[0], tar_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        x1 = self.downConv1(x)
        # print("X:", x.shape, "=>", x1.shape)
        x2 = self.downConv2(x1)
        # print("X:", x1.shape, "=>", x2.shape)
        x3 = self.downConv3(x2)
        # print("X:", x2.shape, "=>", x3.shape)
        x4 = self.downConv4(x3)
        # print("X:", x3.shape, "=>", x4.shape)

        # decoder
        x5 = self.upConv1(x4)
        x5 = torch.cat([x5, x3], dim=1)

        x6 = self.upConv2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.upsample(x6)

        x7 = self.upConv3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.upsample(x7)

        x8 = self.upConv4(x7)
        pred = self.convPred(x8)
        return pred


class BergVGG(nn.Module):
    """
    Modified Berg's methods using VGG11 pretrains
    VGG source code from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    """
    def __init__(self, src_ch, tar_ch, bn=False):
        super(BergVGG, self).__init__()
        self.src_ch = src_ch
        kernels = [64, 128, 256, 512]
    
        self.downConv1 = self.make_layers(3, [64, 'M'], bn)

        self.downConv2 = self.make_layers(64, [128, 'M'], bn)

        self.downConv3 = self.make_layers(128, [256], bn)

        self.downConv4 = self.make_layers(256, [512], bn)

        self.upConv1 = nn.Sequential(
            nn.Conv2d(kernels[3], kernels[3], 3, padding=1),
            # nn.BatchNorm2d(kernels[3]),
            nn.Dropout2d(p=0.5, inplace=True),
            nn.ReLU(inplace=True),)

        self.upConv2 = nn.Sequential(
            nn.Conv2d(kernels[3]+kernels[2], kernels[2], 3, padding=1),
            # nn.BatchNorm2d(kernels[2]),
            nn.Dropout2d(p=0.5, inplace=True),
            nn.ReLU(inplace=True),)

        self.upConv3 = nn.Sequential(
            nn.Conv2d(kernels[2]+kernels[1], kernels[1], 3, padding=1),
            # nn.BatchNorm2d(kernels[1]),
            nn.Dropout2d(p=0.5, inplace=True),
            nn.ReLU(inplace=True),)

        # blue in paper
        self.upConv4 = nn.Sequential(
            nn.Conv2d(kernels[1]+kernels[0], kernels[0], 3, padding=1),
            # nn.BatchNorm2d(kernels[0]),
            # nn.Dropout2d(p=0.5, inplace=True),
            nn.ReLU(inplace=True),)

        self.convPred = nn.Sequential(
            nn.Conv2d(kernels[0], tar_ch, 3, padding=1),
            nn.ReLU(inplace=True),)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(in_channels, cfg, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.src_ch == 1:
            x = torch.cat([x, x, x], dim=1)
        # encoder
        x1 = self.downConv1(x)
        # print("X1:", x.shape, "=>", x1.shape)
        x2 = self.downConv2(x1)
        # print("X2:", x1.shape, "=>", x2.shape)
        x3 = self.downConv3(x2)
        # print("X3:", x2.shape, "=>", x3.shape)
        x4 = self.downConv4(x3)
        # print("X4:", x3.shape, "=>", x4.shape)

        # decoder
        x5 = self.upConv1(x4)
        x5 = torch.cat([x5, x3], dim=1)

        x6 = self.upConv2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.upsample(x6)

        x7 = self.upConv3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.upsample(x7)

        x8 = self.upConv4(x7)
        pred = self.convPred(x8)
        return pred


def berg(src_ch=3, tar_ch=3, pretrained=False, batch_norm=False, **kwargs):
    """Constructs a Berg model.
    Berg, Amanda, Jorgen Ahlberg, and Michael Felsberg. \
    "Generating visible spectrum images from thermal infrared." \
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.
    """
    if isinstance(tar_ch, list):
        tar_ch = sum(tar_ch)
    model = Berg(src_ch, tar_ch)
    return model


def bergVGG(src_ch=3, tar_ch=3, pretrained=False, batch_norm=False, **kwargs):
    """Constructs a Berg model.
    Berg, Amanda, Jorgen Ahlberg, and Michael Felsberg. \
    "Generating visible spectrum images from thermal infrared." \
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.
    """
    if isinstance(tar_ch, list):
        tar_ch = sum(tar_ch)
    model = BergVGG(src_ch, tar_ch)
    if pretrained:
        print("Loading imagenet weights")
        from collections import OrderedDict
        pretrained_state = model_zoo.load_url(model_urls['vgg11_bn'])
        model_state = model.state_dict()
        selected_state = OrderedDict()
        pre_items = list(pretrained_state.items())
        m_items = list(model_state.items())
        for idx in range(min(len(m_items), len(pre_items))):
            pre_k, pre_v = pre_items[idx]
            m_k, m_v = m_items[idx]
            if pre_v.size() == m_v.size():
                # print("Adding {} => {}, size : {}".format(pre_k, m_k, m_v.size()))
                selected_state[m_k] = pre_v
        model_state.update(selected_state)
        model.load_state_dict(model_state)
    return model


if __name__ == "__main__":
    # Hyper Parameters
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-img_row', type=int, default=224,
                        help='img_row of input')
    parser.add_argument('-img_col', type=int, default=224,
                        help='img_col of input ')
    parser.add_argument('-src_ch', type=int, default=1,
                        help='nb channel of source')
    parser.add_argument('-tar_ch', type=int, default=3,
                        help='nb channel of target')
    parser.add_argument('-batch_size', type=int, default=12,
                        help='batch_size for training ')
    args = parser.parse_args()

    x = torch.FloatTensor(
        np.random.random((args.batch_size, args.src_ch, args.img_row, args.img_col)))

    generator = berg(args.src_ch, args.tar_ch)
    gen_y = generator(x)
    total_params = sum(p.numel() for p in generator.parameters())
    print("Berg")
    print(" Network output : ", gen_y.shape)
    print(" Params: {:0.1f}M".format(total_params / (10**6)))

    generator = bergVGG(args.src_ch, args.tar_ch, True)
    gen_y = generator(x)
    total_params = sum(p.numel() for p in generator.parameters())
    print("BergVGG")
    print(" Network output : ", gen_y.shape)
    print(" Params: {:0.1f}M".format(total_params / (10**6)))