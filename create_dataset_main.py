#coding=utf-8
'''images -->lmdb'''
import os
import glob
import argparse
from create_dataset import *

def get_args():
    '''get args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', default='data/lmdb/train')
    parser.add_argument('--data_path', default='data/train')
    parser.add_argument('--suffix', default='*.jpg')
    return parser.parse_args()

def main(args):
    '''main'''
    if not os.path.exists(args.lmdb_path):
        os.makedirs(args.lmdb_path)
    imgpathlist = []
    labellist = []

    for filename in glob.glob(os.path.join(args.data_path, args.suffix)):
        linepart = filename.split('.')[0].split('_')
        if len(linepart) < 2:
            continue
        imgpathlist.append(filename)
        labellist.append(linepart[-1])
        print filename, linepart[-1]
    createDataset(args.lmdb_path, imgpathlist, labellist)

if __name__ == '__main__':
    main(get_args())
