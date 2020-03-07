

import os
import argparse
import numpy as np
import cv2
import json


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as fi:
        content = json.loads(fi.read())
        print(json_path, type(content), content)
        words_num = content['words_result_num']
        words_result = content['words_result']
        inner_rect = np.array(content['inner_rect']['points'], dtype=np.float32).reshape(-1, 2)
        outer_rect = np.array(content['outer_rect']['points'], dtype=np.float32).reshape(-1, 2)
        print(inner_rect)
        print(outer_rect)
        print(words_num)
        print(words_result)
    return words_result, inner_rect, outer_rect


def crop_quad(img, q):
    dst_width = int(round((q[1][0] + q[2][0] - q[0][0] - q[3][0]) / 2))
    dst_height = int(round((q[2][1] + q[3][1] - q[0][1] - q[1][1]) / 2))
    print(q, dst_width, dst_height)
    rect_dst = np.array([[0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, rect_dst)
    img_dst = cv2.warpPerspective(img, M, (dst_width, dst_height))
    print(img_dst.shape)
    return img_dst


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='D:\\data_md\\duxiaogang\\5-4-1218\\img')
    parser.add_argument('--json_dir', default='D:\\data_md\\duxiaogang\\5-4-1218\\json')
    parser.add_argument('--out_dir', default='D:\\data_md\\duxiaogang\\5-4-1218_text-line\\')
    return parser.parse_args()


def main(args):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    for i, filename in enumerate(os.listdir(args.img_dir)):
        img_path = os.path.join(args.img_dir, filename)
        basename = os.path.splitext(filename)[0]
        json_path = os.path.join(args.json_dir, basename + '.json')
        if filename[0] == '.' or not os.path.isfile(img_path) or not os.path.isfile(json_path):
            print(i, filename, img_path, json_path)
            continue
        words_result, inner_rect, outer_rect = read_json(json_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_inner = crop_quad(img, inner_rect)
        img_outer = crop_quad(img, outer_rect)
        inner_path = os.path.join(args.out_dir, basename + '.inner.jpg')
        outer_path = os.path.join(args.out_dir, basename + '.outer.jpg')
        cv2.imwrite(inner_path, img_inner)
        cv2.imwrite(outer_path, img_outer)

        for j, x in enumerate(words_result):
            ts, pts = x['transcription'], x['points']
            pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
            img_text = crop_quad(img, pts)
            text_path = os.path.join(args.out_dir, '{}.{}.jpg'.format(basename, j))
            cv2.imwrite(text_path, img_text)
        break


if __name__ == '__main__':
    main(get_args())