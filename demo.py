import torch
import utils
import dataset
from PIL import Image
from data_generator.config import Alphabet
import models.crnn as crnn

model_path = './output/netCRNN_0.pth'
img_path = '/data/datasets/segment-free/test/0_song5_0_3_b_w.jpg'
alphabet = Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = crnn.CRNN(32, 1, len(alphabet), 256).to(device)
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((200, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size()).to(device)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = torch.IntTensor([preds.size(0)])
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))