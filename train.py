from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.utils.data
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
from models import crnn
from data_generator.config import Alphabet
import pandas

def val(net, test_loader, criterion, converter, device):
    net.eval()
    n_correct = 0
    val_loss = 0.0

    for i, (images, labels) in enumerate(test_loader):
        batch_size = images.size(0)
        text, length = converter.encode(labels)
        text = torch.Tensor(text).int().to(device)
        length = torch.Tensor(length).int().to(device)
        images, texts = images.to(device)

        preds = net(images)
        preds_size = torch.Tensor([preds.size(0)] * batch_size).int()
        loss = criterion(preds, text, preds_size, length) / batch_size
        val_loss += loss

        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, texts):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    pandas_show = [[raw_pred, pred, gt] for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels)]
    print(pandas.DataFrame(data=pandas_show, columns=['network_output', 'ctc_output', 'ground_truth']))

    accuracy = n_correct / float(test_loader.dataset.__len__() * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (val_loss, accuracy))


def train(opt):
    if opt.output_dir is None:
        opt.output_dir = 'expr'
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    device = torch.device("cuda:{0}".format(opt.gpu) if opt.gpu is not None and torch.cuda.is_available() else "cpu")
    train_dataset = dataset.lmdbDataset(root=opt.trainroot)
    assert train_dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                                               num_workers=int(opt.workers),
                                               collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW,
                                                                               keep_ratio=opt.keep_ratio))
    test_dataset = dataset.lmdbDataset(root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=opt.batchSize,
                                              num_workers=int(opt.workers))
    nclass = len(opt.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(opt.alphabet)
    criterion = CTCLoss()

    # custom weights initialization called on crnn
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    net = crnn.CRNN(opt.imgH, nc, nclass, opt.nh).to(device)
    net.apply(weights_init)
    if opt.crnn != '':
        print('loading pretrained model from %s' % opt.crnn)
        net.load_state_dict(torch.load(opt.crnn))

    net = net.to(device)
    criterion = criterion.cuda()

    # setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    for epoch in range(opt.epochs):
        train_loss = 0.0
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            batch_size = images.size(0)
            text, length = converter.encode(labels)

            text = torch.Tensor(text).int().to(device)
            length = torch.Tensor(length).int().to(device)
            image = images.to(device)

            preds = net(image)
            preds_size = torch.Tensor([preds.size(0)] * batch_size).int()
            loss = criterion(preds, text, preds_size, length) / batch_size
            net.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' % (epoch, opt.epochs, i, len(train_loader), train_loss))
        val(net, test_loader, criterion, converter, device)
        torch.save(net.state_dict(), '{0}/netCRNN_{1}.pth'.format(opt.output_dir, epoch))
    torch.save(net, opt.output_dir + 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainroot', default='/data/datasets/segment-free/lmdb', help='path to dataset')
    parser.add_argument('--valroot', default='/data/datasets/segment-free/lmdb', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--gpu', default='0', help='GPU to use')
    parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
    parser.add_argument('--alphabet', type=str, default=Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED)
    parser.add_argument('--output_dir', default=None, help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
    parser.add_argument('--random_sample', action='store_true',
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--keep_ratio', default=True, action='store_true',
                        help='whether to keep ratio for image resize')
    opt = parser.parse_args()
    print(opt)
    train(opt)
