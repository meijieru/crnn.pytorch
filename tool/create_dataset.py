# -*- coding: utf-8 -*-
# @Time    : 18-5-20 下午8:07
# @Author  : zhoujun

import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import argparse

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    # print (len(imagePathList) , len(labelList))
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print('...................')
    env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i]).encode()
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def get_path_label(data_label_path):
    imagePathList = []
    labelList = []
    with open(data_label_path) as t:
        for line in t.readlines():
            line = line.strip('\n')
            line = line.rstrip()
            if len(line)>0:
                params = line.split(' ')
                imagePathList.append(params[0])
                labelList.append(params[1])
    return imagePathList,labelList



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-f','--file',default='/data/datasets/segment-free/test.csv',help='the path of label file')
    args.add_argument('-o','--output',default='/data/datasets/segment-free/lmdb',help='the path of output')
    arg = args.parse_args()

    imagePathList, labelList = get_path_label(arg.file)
    createDataset(arg.output,imagePathList,labelList)