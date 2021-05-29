import torch.nn as nn

from .arch_util import srntt_init_weights


class ImageDiscriminator(nn.Module):

    def __init__(self, in_nc=3, ndf=32):
        super(ImageDiscriminator, self).__init__()

        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True))
            return block

        self.conv_block1 = conv_block(in_nc, ndf)
        self.conv_block2 = conv_block(ndf, ndf * 2)
        self.conv_block3 = conv_block(ndf * 2, ndf * 4)
        self.conv_block4 = conv_block(ndf * 4, ndf * 8)
        self.conv_block5 = conv_block(ndf * 8, ndf * 16)

        self.out_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(ndf * 16, 1024, kernel_size=1),
            nn.LeakyReLU(0.2), nn.Conv2d(1024, 1, kernel_size=1), nn.Sigmoid())

        srntt_init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, x):
        fea = self.conv_block1(x)
        fea = self.conv_block2(fea)
        fea = self.conv_block3(fea)
        fea = self.conv_block4(fea)
        fea = self.conv_block5(fea)

        out = self.out_block(fea)

        return out
