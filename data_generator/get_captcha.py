# -*- coding: utf-8 -*-
# @Time    : 2018/1/7 14:51
# @Author  : zhoujun
from PIL import Image, ImageDraw, ImageFont
import random


class Captcha(object):
    def __init__(self, font_path, font_size=30, size=(100, 40), font_color=(0, 0, 0), background=None):
        self.font = ImageFont.truetype(font_path, font_size)
        self.font_size = font_size
        self.size = size
        self.font_color = font_color
        self.image = self.get_image(background)
        self.bk_image = self.image

    def rotate(self, random_rot=True, rot_angle=None, random_range=(-10, 10)):
        '''
        对字体进行旋转
        :param random_rot: 是否随机旋转
        :param rot_angle: 不随机旋转时给定的角度
        :param random_range: 随机旋转时的角度范围
        :return:None
        '''
        if random_rot:
            rot_angle = random.randint(random_range[0], random_range[1])
        rot = self.image.rotate(rot_angle, expand=0)
        self.image = Image.composite(rot, self.bk_image, rot)

    def get_image(self, background):
        if background:
            background_img = Image.open(background)
            if background_img.size[0] < self.size[0] or background_img.size[1] < self.size[1]:
                return_img = background_img.resize(self.size)
            else:
                random_left = random.randint(0, background_img.size[0] - self.size[0] - 1)
                random_top = random.randint(0, background_img.size[1] - self.size[1] - 1)
                background_img.close()
                return_img = background_img.crop((random_left, random_top, random_left + self.size[0], random_top + self.size[1]))
            background_img.close()
            return return_img
        else:
            return  Image.new('RGBA', self.size, (255, 255, 255))

    def rand_color(self):
        self.font_color = (random.randint(0, 250), random.randint(0, 250), random.randint(0, 250))

    def write(self, text, x):
        draw = ImageDraw.Draw(self.image)
        draw.text((x, 0), text, fill=self.font_color, font=self.font)

    def write_texts(self, texts, write_single=False, random_color=False, rotate=False):
        x = 0
        if write_single:
            for text in texts:
                if random_color:
                    self.rand_color()
                self.write(text, x)
                if rotate:
                    self.rotate()
                x += self.font_size
        else:
            self.write(texts, x)
            # self.image = self.image.transform((self.size[0] + 20, self.size[1] + 10), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0),
            #                         Image.BILINEAR)  # 创建扭曲

    def add_noise(self):
        pass

    def save(self, filename):
        self.image.save(filename)


if __name__ == '__main__':
    font_path = r'Z:\zhoujun\tf-crnn\hlp\fonts\0.ttc'
    draw_text = '啊啊啊啊'
    img = Captcha(font_path=font_path, background=None)
    num = img.write_texts(draw_text)
    img.image.show()
