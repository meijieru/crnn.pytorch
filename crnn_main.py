from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
from config import Alphabet
from models import crnn
import pandas


def val(net, dataset, criterion, converter):
    print('Start val')
    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)
    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    for i in range(len(data_loader)):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.squeeze(1)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    pandas_show = [[raw_pred, pred, gt] for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts)]
    print(pandas.DataFrame(data=pandas_show, columns=['network_output', 'ctc_output', 'ground_truth']))

    accuracy = n_correct / float(len(data_loader) * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def train(opt):
    if opt.output_dir is None:
        opt.output_dir = 'expr'
    if not os.path.exists(opt.output_dir):
        os.system('mkdir {0}'.format(opt.output_dir))
    #
    # opt.manualSeed = random.randint(1, 10000)  # fix seed
    # print("Random Seed: ", opt.manualSeed)
    # random.seed(opt.manualSeed)
    # np.random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)
    #
    # cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.gpu:
        print("有可用gpu，请设置gpu")

    train_dataset = dataset.lmdbDataset(root=opt.trainroot)
    assert train_dataset
    if not opt.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=False, sampler=sampler,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
    test_dataset = dataset.lmdbDataset(
        root=opt.valroot, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

    nclass = len(opt.alphabet)
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

    net = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
    net.apply(weights_init)
    if opt.crnn != '':
        print('loading pretrained model from %s' % opt.crnn)
        net.load_state_dict(torch.load(opt.crnn))
    # print(net)

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)

    if opt.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[int(gpu) for gpu in opt.gpu.split(',')])
        image = image.cuda()
        criterion = criterion.cuda()

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def trainBatch(net, criterion, optimizer):
        data = train_iter.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        return cost

    for epoch in range(opt.epochs):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in net.parameters():
                p.requires_grad = True
            net.train()

            cost = trainBatch(net, criterion, optimizer)
            loss_avg.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, opt.epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        val(net, test_dataset, criterion,converter)
        torch.save(net.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.output_dir, epoch, i))
    torch.save(net, opt.output_dir + 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainroot', default='/data/datasets/segment-free/lmdb', help='path to dataset')
    parser.add_argument('--valroot', default='/data/datasets/segment-free/lmdb', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=200, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--gpu', type=str, default='0', help='GPUs to use eg 1,2')
    parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
    parser.add_argument('--alphabet', type=str, default=Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED)
    parser.add_argument('--output_dir', default=None, help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
    parser.add_argument('--random_sample', action='store_true',
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--keep_ratio', default=True, action='store_true',
                        help='whether to keep ratio for image resize')
    opt = parser.parse_args()
    print(opt)
    train(opt)
