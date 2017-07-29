#encoding=utf8
'''
generate images for training
'''
import os
import random
import argparse
import PIL
import PIL.ImageOps
from PIL import Image, ImageDraw, ImageFont

def gen_rand(char_set, num1, num2):
    '''generate random string'''
    buf = ""
    max_len = random.randint(num1, num2)
    for i in range(max_len):
        buf += random.choice(char_set)
    return buf

def get_args():
    '''get args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='train')
    parser.add_argument('--make_num', type=int, default=10000)
    parser.add_argument('--img_w', type=int, default=100)
    parser.add_argument('--img_h', type=int, default=32)
    parser.add_argument('--cnum1', type=int, default=7)
    parser.add_argument('--cnum2', type=int, default=7)
    parser.add_argument('--char_set', default='0123456789')
    parser.add_argument('--font_path', default="/Library/Fonts/华文仿宋.ttf")
    return parser.parse_args()

def main(args):
    '''main'''
    font = ImageFont.truetype(args.font_path, size=28)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for ind in xrange(args.make_num):
        content = gen_rand(args.char_set, args.cnum1, args.cnum2)
        img = Image.new('L', (args.img_w, args.img_h), (255))
        # img = Image.new('RGB', (args.img_w, args.img_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), content, font=font)
        img = PIL.ImageOps.invert(img)
        savepath = os.path.join(args.output, '{:08d}'.format(ind)+'_'+content+'.jpg')
        print savepath, content
        img.save(savepath)

if __name__ == '__main__':
    main(get_args())