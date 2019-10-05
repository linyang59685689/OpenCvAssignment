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


# img = cv2.imread("res/lena512color.tiff", cv2.IMREAD_COLOR)
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


# img = cv2.imread("res/lena512color.tiff", cv2.IMREAD_COLOR)
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
# img = cv2.imread("res/lena512color.tiff", cv2.IMREAD_GRAYSCALE)
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
# img = cv2.imread("res/lena512color.tiff", cv2.IMREAD_COLOR)
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


# img = cv2.imread("res/lena512color.tiff", cv2.IMREAD_COLOR)
# M = getAffineM(img.shape[1], img.shape[0])
# print(M)
# cv2.imshow("img", img)
# img_affine = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow("affine", img_affine)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Perspective Transform   Homography
def getPerspectiveM(w, h):
    pos1 = numpy.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pos2 = numpy.float32([[0, 0], [w, 0], [0, h], [w, h]])
    return cv2.getPerspectiveTransform(pos1, pos2)


#
# img = cv2.imread("res/lena512color.tiff", cv2.IMREAD_COLOR)
# M = getPerspectiveM(img.shape[1], img.shape[0])
# print(M)
# cv2.imshow("img", img)
# img_affine = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow("affine", img_affine)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Image Blending
# img = cv2.imread("res/lena512color.tiff", cv2.IMREAD_COLOR)
# M=cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),180,1.0)
# img2=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
# new_img=cv2.addWeighted(img,0.9,img2,0.1,0)
# cv2.imshow("hehe",new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Bitwise Operations
# img = cv2.imread("res/lena512color.tiff", cv2.IMREAD_COLOR)
# img=cv2.resize(img,(1000,1000))
# logo = cv2.imread("logo.png", cv2.IMREAD_COLOR)
# new_logo = cv2.resize(logo, (logo.shape[1] , logo.shape[0]))
#
# rows, cols, channels = new_logo.shape
# roi = img[0:rows, 0:cols]
# logo2gray = cv2.cvtColor(new_logo, cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(logo2gray, 10, 255, cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
# img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# img2_fg = cv2.bitwise_and(new_logo,new_logo,mask = mask)
# dst = cv2.add(img_bg,new_logo)
# img[0:rows, 0:cols]=dst
# cv2.namedWindow("img",cv2.WINDOW_NORMAL)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Measuring Performance with OpenCV
# img = cv2.imread("res/lena512color.tiff", cv2.IMREAD_COLOR)
# e1 = cv2.getTickCount()
# print(cv2.useOptimized())
# # for i in range(5,49,2):
# img = cv2.medianBlur(img, 3)
# e2 = cv2.getTickCount()
# fre = cv2.getTickFrequency()
# print(e1, e2, fre, (e2 - e1) / fre)
# cv2.imshow("show", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Changing Color-space

# cap = cv2.VideoCapture("res/video_1280x720.mp4")
# while True:
#     _, frame = cap.read()
#     # hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     lower_blue=numpy.array([110,0,0])
#     upper_blue=numpy.array([130,255,255])
#     mask=cv2.inRange(frame,lower_blue,upper_blue)
#     print(mask)
#     res=cv2.bitwise_and(frame,frame,mask=mask)
#     cv2.imshow("frame",frame)
#     cv2.imshow('mask',mask)
#     cv2.imshow('res',res)
#     print(_, frame)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27 or not _:
#         break
# cv2.destroyAllWindows()
