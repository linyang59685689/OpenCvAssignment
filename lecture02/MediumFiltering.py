# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/10/14 下午6:33
# @Author: LiLinYang
# @File  : MediumFiltering.py

import cv2
import time
import numpy
import random


def medium_gray(img, bSize=(3, 3)):
    new_img = numpy.zeros(img.shape, numpy.uint8)
    img_width = img.shape[1]
    img_height = img.shape[0]

    for i in range(img_width):
        for j in range(img_height):
            if i < bSize[0] // 2 or i >= img_width - bSize[0] // 2:
                new_img[i] = img[i]
                break
            elif j < bSize[1] // 2 or j >= img_height - bSize[1] // 2:
                new_img[:][j] = img[:][j]
            else:
                new_img[i][j] = help_sort(img, i - bSize[0] // 2, j - bSize[1] // 2,
                                          i + bSize[0] // 2,
                                          j + bSize[1] // 2)
    return new_img


def help_sort(img, left, top, right, bottom):
    array = []
    for i in range(right - left + 1):
        for j in range(bottom - top + 1):
            array.append(img[left + i][top + j])
    a = sorted(array)
    return a[len(array) // 2]


def add_salt_noise_gray(img):
    count = img.shape[0] * img.shape[1] // 100
    for i in range(count):
        h = random.randint(0, img.shape[0] - 1)
        w = random.randint(0, img.shape[1] - 1)
        img[h, w] = 255


img = cv2.imread("lena.jpg")
n = img.shape[2]
colors = cv2.split(img)
for i in range(n):
    add_salt_noise_gray(colors[i])
img = cv2.merge(colors)
for i in range(n):
    statr = time.time()
    colors[i] = medium_gray(colors[i], (3, 3))
    print("时间：" + str(time.time() - statr))
new_img = cv2.merge(colors)
# new_img=cv2.medianBlur(img,3)
cv2.imshow('medium', numpy.hstack((img, new_img)))
cv2.waitKey(0)
