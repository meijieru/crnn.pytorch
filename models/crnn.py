import torch.nn as nn
import torch.nn.parallel


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, ngpu, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        self.ngpu = ngpu
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2, 2),
                                                            (2, 2)))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2, 2),
                                                            (2, 2)))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2),
                                                            (2, 2)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2),
                                                            (2, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(nn.LSTM(512, nh, n_rnn, bidirectional=True))
        self.text = nn.Sequential(nn.Linear(nh * 2, nclass))  # [T, b, nclass]

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        # conv features
        conv = nn.parallel.data_parallel(self.cnn, input, gpu_ids)

        # rnn features
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        recurrent, _ = nn.parallel.data_parallel(self.rnn, conv,
                                                 gpu_ids)  # [T, b, h * 2]

        # text classifier
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        text = nn.parallel.data_parallel(self.text, t_rec,
                                         gpu_ids)  # [T * b, nclass]
        text = text.view(T, b, -1)

        return text
