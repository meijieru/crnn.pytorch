import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = './expr/netCRNN_24_6200.pth'
img_path = './data/732.jpg'
#alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
alphabet = '0123456789'
model = crnn.CRNN(32, 1, 11, 256)
if torch.cuda.is_available():
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(1))
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
