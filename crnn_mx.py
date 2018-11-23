# -*- coding: utf-8 -*-
# @Time    : 18-11-16 下午5:46
# @Author  : zhoujun
import torch
import torchvision
from torchvision.models.densenet import _DenseBlock
from torch import nn


class VGG(nn.Module):
    def __init__(self, in_channels):
        super(VGG, self).__init__()
        self.features = nn.Sequential(  # conv layer
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # second conv layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # third conv layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # fourth conv layer
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # fifth conv layer
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            # sixth conv layer
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # seren conv layer
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):
        return self.features(x)


class BasicBlockV2(nn.Module):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, in_channels, out_channels, stride, downsample=False):
        super(BasicBlockV2, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )
        if downsample:
            self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv(x)

        return x + residual


class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlockV2(in_channels=64, out_channels=64, stride=1, downsample=True),
            BasicBlockV2(in_channels=64, out_channels=128, stride=1, downsample=True),
            nn.Dropout(0.2),

            BasicBlockV2(in_channels=128, out_channels=128, stride=2, downsample=True),
            BasicBlockV2(in_channels=128, out_channels=256, stride=1, downsample=True),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False),

            BasicBlockV2(in_channels=256, out_channels=512, stride=1, downsample=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=2, padding=(0, 1), bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels, pool_stride, pool_pad, dropout):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        if dropout:
            self.add_module('dropout', nn.Dropout(dropout))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=pool_stride, padding=pool_pad))


class DenseNet(nn.Module):
    def __init__(self, in_channels):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),
            _DenseBlock(num_input_features=64, num_layers=8, bn_size=4, growth_rate=8, drop_rate=0),
            _Transition(in_channels=128, out_channels=128, pool_stride=2, pool_pad=0, dropout=0.2),

            _DenseBlock(num_input_features=128, num_layers=8, bn_size=4, growth_rate=8, drop_rate=0),
            _Transition(in_channels=192, out_channels=192, pool_stride=(2, 1), pool_pad=(0, 1), dropout=0.2),

            _DenseBlock(num_input_features=192, num_layers=8, bn_size=4, growth_rate=8, drop_rate=0),

            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, padding=(0, 1), bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)


class BidirectionalLSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        # x = self.fc(x)  # [T * b, nOut]
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.cnn = ResNet(in_channels=in_channels)  # DenseNet()  # VGG()

    def forward(self, x):
        return self.cnn(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, n_class, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(in_channels=in_channels, hidden_size=hidden_size, num_layers=num_layers),
            BidirectionalLSTM(in_channels=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers))
        self.fc = nn.Linear(hidden_size * 2, n_class)

    def forward(self, x):
        x = self.rnn(x)
        x = self.fc(x)
        return x


class CRNN(nn.Module):
    def __init__(self, in_channels, n_class, hidden_size=256, num_layers=1):
        super(CRNN, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(2048, n_class, hidden_size, num_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = x.squeeze(dim=2)
        x = x.permute(2, 0, 1)  # [w, b, c]
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    a = torch.zeros((2, 3, 32, 320)).to(device)
    net = CRNN(3, 10, 512)
    # net = VGG(3)
    # net.hybridize()
    net.to(device)
    b = net(a)
    print(b.size())