# -*- coding: utf-8 -*-
# You can save all generated images in a single h5py file.
# I just want to have a look with my generated pics directly.
from functools import reduce

from PIL import Image
import random
import os
import argparse
from tqdm import tqdm
import numpy as np
import skimage
import csv
import cv2
from get_captcha import Captcha
from config import Alphabet


def matrix2image(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    new_im = new_im.convert('RGB')
    return new_im


def process_str(x):
    return ''.join([char + ' ' if char.isnumeric() else char for char in x])


def write_one(pic_height, to_print_text, fonts, pic_dir, m_count):
    '''
    根据输入参数，生成一张图像并且保存
    :param pic_height: 图像高度
    :param to_print_text: 图像上的文字
    :param fonts: 使用的字体文件名
    :param pic_dir: 图像保存路径
    :param m_count: 当前是第几张
    :return: 生成的图像路径和图像上文字标签
    '''
    # 统计文本里面数字的个数
    alpha_numeric_number_counter = sum(map(lambda x: 1 if x.encode('utf-8').isdigit() else 0, to_print_text))
    # 计算图像宽度
    pic_width = len(to_print_text) * FONT_SIZE + BORDER_SIZE * 2 - int(alpha_numeric_number_counter * 0.5 * FONT_SIZE)
    list_paths = []
    list_labels = []
    for i in range(len(fonts)):
        try:
            img_Captcha = Captcha(font_path=os.path.join(font_dir, fonts[i]), font_size=FONT_SIZE,
                                  size=(pic_width, pic_height), background=None)
            img_Captcha.write_texts(to_print_text)
            img = img_Captcha.image
            if fonts[i].__contains__('msyh'):
                img = img.crop((0, int(BORDER_SIZE*2), pic_width, pic_height - int(BORDER_SIZE*0)))
            elif fonts[i].__contains__('FZJH'):
                img = img.crop((0, int(BORDER_SIZE), pic_width, pic_height - int(BORDER_SIZE*2.5)))
            else:
                img = img.crop((0, 0, pic_width, pic_height - int(BORDER_SIZE*3)))

            file_name = os.path.join(pic_dir, '%d_%s' % (m_count, fonts[i][:-4]))
            # 随机增加黑点噪声
            img = np.array(img.convert('L'))
            if random.randint(0, 6) > 3:
                img = skimage.util.random_noise(img, mode='pepper', amount=0.002)
                img = (img * 255).astype(np.uint8)
                file_name += '_b'
            # 随机增加白点噪声
            if random.randint(0, 6) > 3:
                img = skimage.util.random_noise(img, mode='salt', amount=0.005)
                img = (img * 255).astype(np.uint8)
                file_name += '_w'
            file_name += '.jpg'
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imwrite(file_name,img)
            list_paths.append(os.path.abspath(file_name))
            list_labels.append(to_print_text)
        except Exception as e:
            print(e)
    return list_paths, list_labels


def write2file_callback(label_file_path, list_paths, list_labels):
    with open(label_file_path, encoding='utf8', mode='a+') as to_write:
        csvwriter = csv.writer(to_write, delimiter=' ')
        for i in range(len(list_paths)):
            csvwriter.writerow([list_paths[i], list_labels[i]])
        # to_write.writelines(lines)
        # to_write.flush()


def write(count: int, mode: str, min_len: int, max_len: int, shape: tuple, worker_num: int):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    pic_dir = os.path.join(data_dir, mode)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    label_file_path = os.path.join(data_dir, mode + '.csv')
    if os.path.exists(label_file_path):
        os.remove(label_file_path)
    list_paths = []
    list_labels = []
    pbar = tqdm(total=count)
    for m_count in range(count):
        pic_height = FONT_SIZE + BORDER_SIZE * 3
        # pic_height = FONT_SIZE
        # 每次将字体乱序
        idx_font = list(range(len(fonts)))
        random.shuffle(idx_font)
        for font_i in idx_font:
            to_print = [text[random.randint(0, text_len - 1)] for __ in range(random.randint(min_len, max_len))]
            to_print_text = ''.join(to_print)
            to_font = [fonts[font_i]]
            list_paths1, list_labels1 = write_one(pic_height, to_print_text, to_font, pic_dir, m_count)
            list_paths.extend(list_paths1)
            list_labels.extend(list_labels1)

        pbar.update(1)
        if (m_count + 1) % 100 == 0 or (m_count + 1) == count:
            write2file_callback(label_file_path, list_paths, list_labels)
            list_paths = []
            list_labels = []
    pbar.close()
    print('finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--name', default='val1', help='the dataset name')
    parser.add_argument('-num', '--num', default=1, type=int, help='how many do you want generate')
    parser.add_argument('-data_dir', '--data_dir', default='', help='where to store the generated images')
    parser.add_argument('-font_dir', '--font_dir', default='font', help='the fonts folder location')
    parser.add_argument('-font_size', '--font_size', default=32, type=int, help='the font size')
    parser.add_argument('-border_size', '--border_size', default=6, type=int, help='the border of text images')
    parser.add_argument('-trd', '--trd', default=0, type=int,
                        help='text rotate degree limit(clockwise & counter clockwise)')
    parser.add_argument('-charset', '--charset', default='dict.txt', help='location of charset file,only one line!!!')
    parser.add_argument('-min_len', '--min_len', default=3, type=int,
                        help='the min length of text in generated images ')
    parser.add_argument('-max_len', '--max_len', default=10, type=int,
                        help='the max length of text in generated images ')
    parser.add_argument('-width', '--width', default=200, type=int, help='the width of the generated images')
    parser.add_argument('-height', '--height', default=32, type=int, help='the height of the generated images')
    parser.add_argument('-shuffle', '--shuffle', action='store_true', help='shuffle the csv')
    parser.add_argument('-shuffle_count', '--shuffle_count', default=10000, type=int, help='shuffle the csv')
    parser.add_argument('-worker', '--worker', default=1, type=int,
                        help='use multiprocess to speed up image generate(num of CPU cores minus 1 is RECOMMEND)')
    opt = parser.parse_args()
    print(opt)

    data_dir = opt.data_dir
    font_dir = opt.font_dir
    FONT_SIZE = opt.font_size
    BORDER_SIZE = opt.border_size
    TEXT_ROTATE_DEGREE = opt.trd
    shuffle_count = opt.shuffle_count

    fonts = [_ for _ in os.listdir(font_dir)]
    # 读取词典文本
    # with open(opt.charset, encoding='utf8') as to_read:
    # text = ''.join(map(str.strip, to_read.readlines()))
    text = Alphabet.CHINESECHAR_LETTERS_DIGIT # 生成数据使用的字符不应该带有空白字符
    text_len = len(text)
    print('using characters', text_len, 'using fonts', len(fonts))
    # 开始生成数据
    write(opt.num, opt.name, opt.min_len, opt.max_len, (opt.width, opt.height), opt.worker)
    # write_one(FONT_SIZE + BORDER_SIZE * 2,'其他情况',fonts,'./',0)
