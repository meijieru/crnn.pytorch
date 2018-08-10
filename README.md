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
Train a new model
-----------------
1. Construct dataset following origin guide. For training with variable length, please sort the image according to the text length.
2. ``python crnn_main.py [--param val]``. Explore ``crnn_main.py`` for details.
