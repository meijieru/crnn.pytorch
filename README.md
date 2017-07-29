Convolutional Recurrent Neural Network
======================================

This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)

Run demo
--------
A demo program can be found in ``src/demo.py``. Before running the demo, download a pretrained model
from [Baidu Netdisk](https://pan.baidu.com/s/1pLbeCND) or [Dropbox](https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0). 
This pretrained model is converted from auther offered one by ``tool``.
Put the downloaded model file ``crnn.pth`` into directory ``data/``. Then launch the demo by:

    python demo.py

The demo reads an example image and recognizes its text content.

Example image:
![Example Image](./data/demo.png)

Expected output:
    loading pretrained model from ./data/crnn.pth
    a-----v--a-i-l-a-bb-l-ee-- => available

Dependence
----------
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
* lmdb

Train a new model
-----------------
1. Construct dataset following origin guide. For training with variable length, please sort the image according to the text length.
```
python gen_image.py --output data/train --make_num 10000
python gen_image.py --output data/val --make_num 1000

python create_dataset_main.py --lmdb_path data/lmdb/train --data_path data/train
python create_dataset_main.py --lmdb_path data/lmdb/val --data_path data/val
```
2. ``python crnn_main.py [--param val]``. Explore ``crnn_main.py`` for details.

```
python crnn_main.py \
    --alphabet='0123456789' \
    --trainroot='data/lmdb/train' \
    --valroot='data/lmdb/val' \
    --workers=1 \
    --batchSize=16 \
    --displayInterval=100 \
    --valInterval=100 \
    --adadelta \
    --lr=0.01 \
    --random_sample
```