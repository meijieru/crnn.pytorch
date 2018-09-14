Convolutional Recurrent Neural Network
======================================

This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)

fork from meijieru/crnn.pytorch https://github.com/meijieru/crnn.pytorch


Envrionment 
--------
python 3.6
pytorch 4.0
opencv2.4 + pytorch + lmdb +wrap_ctc

ATTENTION!
getLmdb.py must run in python2.x

Run demo
--------
A demo program can be found in ``src/demo.py``. Before running the demo, download a pretrained model
from [Baidu Netdisk](https://pan.baidu.com/s/1pLbeCND) or [Dropbox](https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0). 
This pretrained model is converted from auther offered one by ``tool``.
Put the downloaded model file ``crnn.pth`` into directory ``data/``. Then launch the demo by:

    python demo.py

The demo reads an example image and recognizes its text content.

Example image:
![my_example_image](./data/demo.png)

Expected output:
    loading pretrained model from ./data/crnn.pth
    a-----v--a-i-l-a-bb-l-ee-- => available

Dependence
----------
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
* lmdb

error when install warp_ctc_pytorch
----------
* [ 11%] Building NVCC (Device) object CMakeFiles/warpctc.dir/src/warpctc_generated_reduce.cu.o
 sh: cicc: command not found
  CMake Error at warpctc_generated_reduce.cu.o.cmake:279 (message):
  Error generating file
  /home/rice/warp-ctc/build/CMakeFiles/warpctc.dir/src/./warpctc_generated_reduce.cu.o
 make[2]: *** [CMakeFiles/warpctc.dir/build.make:256: CMakeFiles/warpctc.dir/src/warpctc_generated_reduce.cu.o] Error 1
 make[1]: *** [CMakeFiles/Makefile2:104: CMakeFiles/warpctc.dir/all] Error 2
 make: *** [Makefile:130: all] Error 2               you should reinstall your cuda, and make sure it install completely
* THCudaMallco error      https://github.com/baidu-research/warp-ctc/pull/71/files
* https://github.com/Xtra-Computing/thundersvm/issues/54#issuecomment-416413155

Train a new model
-----------------
Construct dataset following origin guide. For training with variable length, please sort the image according to the text length. reference:https://github.com/Aurora11111/TextRecognitionDataGenerator

1. 数据预处理

运行`/contrib/crnn/tool/getLmdb.py`

    # 生成的lmdb输出路径
    outputPath = "./train_lmdb"
    # 图片及对应的label
    imgdata = open("./train.txt")

2. 训练模型

运行`/contrib/crnn/crnn_main.py`

    python crnn_main.py [--param val]
    --trainroot        训练集路径
    --valroot          验证集路径
    --workers          CPU工作核数, default=2
    --batchSize        设置batchSize大小, default=64
    --imgH             图片高度, default=32
    --nh               LSTM隐藏层数, default=256
    --niter            训练回合数, default=25
    --lr               学习率, default=0.01
    --beta1             
    --cuda             使用GPU, action='store_true'
    --ngpu             使用GPU的个数, default=1
    --crnn             选择预训练模型
    --alphabet         设置分类
    --Diters            
    --experiment        模型保存目录
    --displayInterval   设置多少次迭代显示一次, default=500
    --n_test_disp        每次验证显示的个数, default=10
    --valInterval        设置多少次迭代验证一次, default=500
    --saveInterval       设置多少次迭代保存一次模型, default=500
    --adam               使用adma优化器, action='store_true'
    --adadelta           使用adadelta优化器, action='store_true'
    --keep_ratio         设置图片保持横纵比缩放, action='store_true'
    --random_sample      是否使用随机采样器对数据集进行采样, action='store_true'
    
示例:python /contrib/crnn/crnn_main.py --tainroot [训练集路径] --valroot [验证集路径] --nh 128 --cuda --crnn [预训练模型路径] 

修改`/contrib/crnn/keys.py`中`alphabet = '012346789'`增加或者减少类别

3. 注意事项

训练和预测采用的类别数和LSTM隐藏层数需保持一致
