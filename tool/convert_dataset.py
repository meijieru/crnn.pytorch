#!/usr/bin/python
# encoding: utf-8

import sys
origin_path = sys.path
sys.path.append("..")
import dataset
sys.path = origin_path
import lmdb

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

def convert(originPath, outputPath):
    args = [0] * 6
    originDataset = dataset.lmdbDataset(originPath, 'abc', *args)
    print('Origin dataset has %d samples' % len(originDataset))

    labelStrList = []
    for i in range(len(originDataset)):
        label = originDataset.getLabel(i + 1)
        labelStrList.append(label)
        if i % 10000 == 0:
            print(i)

    lengthList = [len(s) for s in labelStrList]
    items = zip(lengthList, range(len(labelStrList)))
    items.sort(key=lambda item: item[0])

    env = lmdb.open(outputPath, map_size=1099511627776)

    cnt = 1
    cache = {}
    nSamples = len(items)
    for i in range(nSamples):
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        origin_i = items[i][1]
        img, label = originDataset[origin_i + 1]
        cache[labelKey] = label
        cache[imageKey] = img
        if cnt % 1000 == 0 or cnt == nSamples:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Convert dataset with %d samples' % nSamples)

if __name__ == "__main__":
    convert('/share/datasets/scene_text/Synth90k/synth90k-val-lmdb', '/share/datasets/scene_text/Synth90k/synth90k-val-ordered-lmdb')
    convert('/share/datasets/scene_text/Synth90k/synth90k-train-lmdb', '/share/datasets/scene_text/Synth90k/synth90k-train-ordered-lmdb')
