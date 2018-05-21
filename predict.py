import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from config import Alphabet
import time

model_path = 'model.pkl'
img_path = '/data/datasets/segment-free/test/0_song5_0_3_b_w.jpg'
alphabet = Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED

model = torch.load(model_path)
if torch.cuda.is_available():
    model = model.cuda()

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()

start = time.time()
preds = model(image)
_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred),time.time()-start)



