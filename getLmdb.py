# -*- coding: utf-8 -*-
import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob

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
        for k, v in cache.iteritems():
            txn.put(k, v)


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
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print('...................')
    # map_size=1099511627776 定义最大空间是1TB
    env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        ##########
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def read_text(path):
    with open(path) as f:
        text = f.read()
    text = text.strip()

    return text


if __name__ == '__main__':

    outputPath = '/run/media/rice/DATA/lmdb'
    imgdata = open("/run/media/rice/DATA/labellist1.txt")
    imagePathList = []
    imgLabelLists = []
    i = 0
    # # for filename in glob.glob(os.path.join('/run/media/rice/DATA/number/data/', '*.jpg')):
    # #     print filename
    # #     imagePathList.append(filename)

    # print len(imagePathList)
    for line in list(imgdata):
        print line
        if i< 1251067:
            label = line.split()[1]
            image = line.split()[0]
            imgLabelLists.append(label)
            imagePathList.append('/run/media/rice/DATA/datasets2/' + image+".jpg")
        else:
            break
        i += 1
    #imgLabelLists = sorted(imgLabelLists, key=lambda x: len(x[0]))

    print len(imagePathList)
    print  len(imgLabelLists)
    createDataset(outputPath, imagePathList, imgLabelLists, lexiconList=None, checkValid=True)


