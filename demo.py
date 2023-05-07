import os
import torch
#from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = './data/crnn.pth'
# 这里在原有的demo基础之上增加了多增加一张手写12345的图片
img_path_list = ['./data/demo1.png', './data/demo.png']
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
if os.path.exists(model_path):
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)
# python的PIL image的resize方法传入的是(width, height)
transformer = dataset.resizeNormalize((100, 32))
image_list = []
for img_path in img_path_list:
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    image = image.unsqueeze(0)
    image_list.append(image)
image = torch.cat(image_list, 0)
if torch.cuda.is_available():
    image = image.cuda()

model.eval()
preds = model(image)

# 这里的preds的大小是(26,1,37)，相当于将图片的在宽的方向上切分26个时间，每个时刻上会预测字符在37个字符上概率分布

_, preds = preds.max(2) # 这里其实是取用了第三维最大值的位置，即概率最大的字符位置
preds = preds.transpose(1, 0).contiguous()

for idx, preds in enumerate(preds):
    preds_size = torch.IntTensor([preds.size(0)])
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True) # 这里是将空白符号也会解码出来的
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False) # 这里使用大beta函数来做映射
    print('%-20s => %-20s' % (raw_pred, sim_pred))
