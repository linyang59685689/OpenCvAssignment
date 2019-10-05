# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/10/1 上午10:37
# @Author: LiLinYang
# @File  : HomeWork.py

import cv2
import numpy as np
import random


# perspective translation
# 图像矫正
def image_correct_test():
    img = cv2.imread("perspective_4000_2250.jpg")
    # 这里是人工找了四个点，期待毕业以后可以不用人工找点，完全自动实现这个功能
    pos1 = np.float32([[533, 199], [3237, 235], [278, 2224], [3476, 2199]])
    pos2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
    M = cv2.getPerspectiveTransform(pos1, pos2)
    img2 = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.imshow("img2", img2)
    # cv2.imwrite("correct.jpg",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def random_translation_color(img, min, max):
    B, G, R = cv2.split(img)

    def translation(src):
        translate = random.randint(min, max)
        print(translate)
        # if translate > 0:
        #     limit = 255 - translate
        #     src[src >= limit] = 255
        #     src[src < limit] = (src[src < limit] + translate)
        # elif translate < 0:
        #     limit = - translate
        #     src[src <= limit] = 0
        #     src[src > limit] = (src[src > limit] + translate)
        cv2.add(src, translate, src)

    translation(B)
    translation(G)
    translation(R)
    return cv2.merge((B, G, R))


def gamma_adjust(img, gamma=1.0):
    # 构建一个表，使用查表法增加计算速度
    table = []
    for i in range(256):
        table.append(((i / 255) ** gamma) * 255)
    table = np.array(table).astype(np.uint8)
    return cv2.LUT(img, table)


img = cv2.imread("lena512color.tiff")
for i in range(5):
    color_img = random_translation_color(img, -50, 50)
    cv2.imwrite("random_color_lena512_" + str(i) + ".jpg", color_img)
    gamma = random.uniform(0.5, 1.5)
    print(gamma)
    gamma_img = gamma_adjust(img, gamma)
    cv2.imwrite("random_gamma_lena512_" + str(i) + ".jpg", gamma_img)
