# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun

from __future__ import print_function
import os
import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data
from warpctc_pytorch import CTCLoss
import utils
from crnn_mx import CRNN
import config
import shutil
from dataset import ImageDataset
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)


def save_checkpoint(checkpoint_path, model, optimizer, epoch):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    print('model loaded from %s' % checkpoint_path)
    return start_epoch


def accuracy(preds, labels, preds_lengths, converter):
    _, preds = preds.max(2)
    preds = preds.squeeze(1)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_lengths.data, raw=False)
    n_correct = 0
    for pred, target in zip(sim_preds, labels):
        if pred == target:
            n_correct += 1
    return n_correct


def evaluate_accuracy(model, dataloader, device, converter):
    model.eval()
    metric = 0
    for i, (images, label) in enumerate(dataloader):
        cur_batch_size = images.size(0)
        images = images.to(device)
        preds = model(images)
        # print(len(images))
        preds_lengths = torch.Tensor([preds.size(0)] * cur_batch_size).int()
        metric += accuracy(preds.cpu(), label, preds_lengths.cpu(), converter)
    return metric


def train():
    torch.random.initial_seed()
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if config.output_dir is None:
        config.output_dir = 'output'
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        print('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        print('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    train_transfroms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(data_txt=config.trainfile, data_shape=(config.img_h, config.img_w), img_type='PIL',
                                 img_channel=config.img_channel, phase='train', transform=train_transfroms)

    train_data_loader = DataLoader(train_dataset, config.train_batch_size, shuffle=True, num_workers=config.workers)

    test_dataset = ImageDataset(data_txt=config.testfile, data_shape=(config.img_h, config.img_w), img_type='PIL',
                                img_channel=config.img_channel, phase='test', transform=transforms.ToTensor())
    test_data_loader = DataLoader(test_dataset, config.eval_batch_size, shuffle=True, num_workers=config.workers)

    converter = utils.strLabelConverter(config.alphabet)
    criterion = CTCLoss()

    model = CRNN(config.img_channel, len(config.alphabet), config.nh)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_decay_step, gamma=config.lr_decay)

    start_epoch = config.start_epoch
    if config.checkpoint != '' and not config.restart_training:
        start_epoch = load_checkpoint(config.checkpoint, model, optimizer)
    model = model.to(device)
    writer = SummaryWriter(config.output_dir)
    # dummy_input = torch.Tensor(1, config.img_channel, config.img_h, config.img_w).to(device)
    # writer.add_graph(model=model, input_to_model=dummy_input)
    all_step = train_dataset.__len__() // config.train_batch_size
    for epoch in range(start_epoch, config.epochs):
        model.train()
        if scheduler.get_lr()[0] > config.end_lr:
            scheduler.step()
        start = time.time()

        batch_acc = .0
        batch_loss = .0
        cur_step = 0
        for i, (images, labels) in enumerate(train_data_loader):
            cur_batch_size = images.size(0)
            targets, targets_lengths = converter.encode(labels)

            targets = torch.Tensor(targets).int()
            targets_lengths = torch.Tensor(targets_lengths).int()
            images = images.to(device)

            preds = model(images)
            preds_lengths = torch.Tensor([preds.size(0)] * cur_batch_size).int()
            loss = criterion(preds, targets, preds_lengths, targets_lengths)  # text,preds_size must be cpu
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item() / cur_batch_size
            acc = accuracy(preds.cpu(), labels, preds_lengths.cpu(), converter) / cur_batch_size
            batch_acc += acc
            batch_loss += loss
            # write tensorboard
            cur_step = epoch * all_step + i
            writer.add_scalar(tag='ctc_loss', scalar_value=loss, global_step=cur_step)
            writer.add_scalar(tag='train_acc', scalar_value=acc, global_step=cur_step)
            writer.add_scalar(tag='lr', scalar_value=scheduler.get_lr()[0], global_step=cur_step)

            if (i + 1) % config.display_interval == 0:
                batch_time = time.time() - start
                # for name, param in model.named_parameters():
                #     if 'bn' not in name:
                #         writer.add_histogram(name, param, cur_step)
                #         writer.add_histogram(name + '-grad', param.grad, cur_step)

                print('[{}/{}], [{}/{}],step: {}, ctc loss:{:.4f}, acc:{:.4f}, lr:{}, time:{:.4f}'.format(
                    epoch, config.epochs, i + 1, all_step, cur_step, batch_loss / config.display_interval,
                                          batch_acc / config.display_interval, scheduler.get_lr()[0], batch_time))
                batch_loss = .0
                batch_acc = .0
                start = time.time()
        print('start eval....')
        # test
        val_acc = evaluate_accuracy(model, test_data_loader, device, converter) / len(test_dataset)
        print('[{}/{}], val_acc: {:.6f}'.format(epoch, config.epochs, val_acc))
        writer.add_scalar(tag='val_acc', scalar_value=val_acc, global_step=cur_step)
        save_checkpoint('{}/{}_{}.pth'.format(config.output_dir, epoch, val_acc), model, optimizer, epoch + 1)


if __name__ == '__main__':
    train()
