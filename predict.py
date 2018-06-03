import torch
import utils
from data_generator.config import Alphabet
import models.crnn as crnn
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname='./data/msyh.ttc')


class Pytorch_model:
    def __init__(self, model_path, alphabet, img_shape, net, img_channel=3, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param alphabet: 字母表
        :param img_shape: 图像的尺寸(w,h)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id
        self.img_w = img_shape[0]
        self.img_h = img_shape[1]
        self.img_channel = img_channel
        self.converter = utils.strLabelConverter(alphabet)
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
            self.net = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
        else:
            self.device = torch.device("cpu")
            self.net = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
        print('device:', self.device)

        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            net.load_state_dict(self.net)
            self.net = net
        self.net.eval()

    def predict(self, img):
        '''
        对传入的图像进行预测，支持图像地址和numpy数组
        :param img: 像地址或numpy数组
        :param is_numpy:
        :return:
        '''
        assert self.img_channel in [1, 3], 'img_channel must in [1.3]'

        if isinstance(img, str):  # read image
            assert os.path.exists(img), 'file is not exists'
            img = cv2.imread(img, 0 if self.img_channel == 1 else 1)

        img = cv2.resize(img, (self.img_w, self.img_h))
        if len(img.shape) == 2 and self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and self.img_channel == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 将图片由(w,h)变为(1,img_channel,h,w)
        img = img.reshape([self.img_h, self.img_w, self.img_channel])
        img = transforms.ToTensor()(img)
        img = img.unsqueeze_(0)

        img = img.to(self.device)
        preds = self.net(img)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = torch.Tensor([preds.size(0)]).int()
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))
        return sim_pred


if __name__ == '__main__':
    model_path = './output/netCRNN_29.pth'
    # model_path = './output/model.pkl'
    img_path = '/data1/zj/dataset/test/0_song5_0_3.jpg'
    alphabet = Alphabet.CHINESECHAR_LETTERS_DIGIT_SYMBOLS
    # 初始化网络
    net = crnn.CRNN(32, 1, len(alphabet), 256)
    model = Pytorch_model(model_path, alphabet=alphabet, net=net, img_shape=[200, 32],
                          img_channel=1)
    # 执行预测
    img = cv2.imread(img_path, 1)
    result = model.predict(img)
    # 可视化
    plt.title(result, fontproperties=myfont, fontsize=18)
    plt.imshow(img, cmap='gray_r')
    plt.show()
