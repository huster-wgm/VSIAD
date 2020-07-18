import argparse
import numpy as np
import torch
import torch.nn as nn


class pixMLP(nn.Module):

    def __init__(self, src_ch=1, tar_ch=1, hidden=64):
        super(pixMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(src_ch, hidden, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden, hidden * 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden * 2, 1, kernel_size=1, stride=1, padding=0, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        return self.mlp(x)


def mlp(src_ch=1, tar_ch=1, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = pixMLP(src_ch, tar_ch)

    return model


if __name__ == "__main__":
    # Hyper Parameters
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-img_row', type=int, default=224,
                        help='img_row of input')
    parser.add_argument('-img_col', type=int, default=224,
                        help='img_col of input ')
    parser.add_argument('-src_ch', type=int, default=2,
                        help='nb channel of source')
    parser.add_argument('-tar_ch', type=int, default=1,
                        help='nb channel of target')
    parser.add_argument('-base_size', type=int, default=12,
                        help='batch_size for training ')
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for discriminator')
    args = parser.parse_args()

    x = torch.FloatTensor(
        np.random.random((args.base_size, args.src_ch, args.img_row, args.img_col)))

    discriminator = mlp(args.src_ch, args.tar_ch)
    pred = discriminator(x)
    total_params = sum(p.numel() for p in discriminator.parameters())
    print("pixMLP =>")
    print(" Network output: ", pred.shape)
    print(" Params: {:0.1f}M".format(total_params / (10**6)))


