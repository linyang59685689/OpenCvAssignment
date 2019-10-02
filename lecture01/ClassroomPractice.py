# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/10/1 上午10:36
# @Author: LiLinYang
# @File  : ClassroomPractice.py

import cv2
from matplotlib import pyplot
import numpy
import random


def random_translation_color(img, min, max):
    B, G, R = cv2.split(img)

    def translation(src):
        translate = random.randint(min, max)
        print(translate)
        if translate > 0:
            limit = 255 - translate
            src[src >= limit] = 255
            src[src < limit] = (src[src < limit] + translate)
        elif translate < 0:
            limit = - translate
            src[src <= limit] = 0
            src[src > limit] = (src[src > limit] + translate)

    translation(B)
    translation(G)
    translation(R)
    return cv2.merge((B, G, R))


# img = cv2.imread("lena512color.tiff", cv2.IMREAD_COLOR)
# print(img.shape)
# new_img = random_translation_color(img, -50, 50)
# cv2.imshow("src", img)
# cv2.imshow("des", new_img)
# cv2.waitKey()
# cv2.destroyAllWindows()


# gamma corection

def gamma_adjust(img, gamma=1.0):
    table = []
    for i in range(256):
        table.append(((i / 255) ** gamma) * 255)
    table = numpy.array(table).astype(numpy.uint8)
    return cv2.LUT(img, table)


# img = cv2.imread("lena512color.tiff", cv2.IMREAD_COLOR)
# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
# new_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
# cv2.imshow("rgb", img)
# cv2.imshow("yuv", new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img=gamma_adjust(img,gamma=0.7)
# print(img.shape)
# cv2.imshow("hehe", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# image crop
# img = cv2.imread("lena512color.tiff", cv2.IMREAD_GRAYSCALE)
# img_crop = img[0:400, 100:512]
# hist = img.flatten()
# pyplot.hist(hist,256,[0,256])
# pyplot.show()
# hist=[355,355,355,355,355,355]
# newImg=cv2.equalizeHist(img)
# cv2.imshow("new",newImg)
# cv2.imshow("src",img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# newhist = newImg.flatten()
# pyplot.hist(newhist,64,[0,256])
# pyplot.show()
# cv2.imshow("eheh", img_crop)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Similarity Transform
# img = cv2.imread("lena512color.tiff", cv2.IMREAD_COLOR)
# M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1.0)
# print(M)
# img2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow("img1", img)
# cv2.imshow("img2", img2)
# print(img.shape)
# print(img2.shape)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Affine Transform
def getAffineM(w, h):
    pos1 = numpy.float32([[0, 0], [w, 0], [0, h]])
    pos2 = numpy.float32([[w, 0], [0, 0], [w, h]])
    return cv2.getAffineTransform(pos1, pos2)

# img = cv2.imread("lena512color.tiff", cv2.IMREAD_COLOR)
# M = getAffineM(img.shape[1], img.shape[0])
# print(M)
# cv2.imshow("img", img)
# img_affine = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow("affine", img_affine)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Perspective Transform   Homography
# def getPerspectiveM(w, h):
#     pos1 = numpy.float32([[0, 0], [w, 0], [0, h],[w,h]])
#     pos2 = numpy.float32([[0, 0], [w, 0], [0, h],[w,h]])
#     return cv2.getPerspectiveTransform(pos1, pos2)
#
# img = cv2.imread("lena512color.tiff", cv2.IMREAD_COLOR)
# M = getPerspectiveM(img.shape[1], img.shape[0])
# print(M)
# cv2.imshow("img", img)
# img_affine = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow("affine", img_affine)
# cv2.waitKey()
# cv2.destroyAllWindows()
