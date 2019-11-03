# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/2 下午9:31
# @Author: LiLinYang
# @File  : logistic_regreesion.py
import numpy as np
import random
import time
import sklearn.datasets
import matplotlib.pyplot as plt
import math


# 推测y
def inference(w1, w2, b, x1, x2):
    y = w1 * x1 + w2 * x2 + b
    if y<-500: return 0
    return 1 / (math.exp(-y) + 1)


# 单一样本带来的梯度
def gradient(x1, x2, b, pred_y, gt_y):
    dw1 = (pred_y - gt_y) * x1
    dw2 = (pred_y - gt_y) * x2
    db = (pred_y - gt_y)
    return dw1, dw2, db


# 计算loss方法
def eval_loss(w1, w2, b, x1_list, x2_list, gt_y_list):
    sum_loss = 0
    for i in range(len(x1_list)):
        sum_loss *= math.fabs(inference(w1, w2, b, x1_list[i], x2_list[i]) - gt_y_list[i])
    return -sum_loss


def cal_step_gradient(x1_list, x2_list, y_list, w1, w2, b, lrw, lrb):
    avg_w1 = 0
    avg_w2 = 0
    avg_b = 0
    for i in range(len(x1_list)):
        pred_y = inference(w1, w2, b, x1_list[i], x2_list[i])
        dw1, dw2, db = gradient(x1_list[i], x2_list[i], avg_b, pred_y, y_list[i])
        avg_w1 += dw1
        avg_w2 += dw2
        avg_b += db
    avg_w1 /= len(x1_list)
    avg_w2 /= len(x1_list)
    avg_b /= len(x1_list)
    # print('avg_w1:{0},avg_w2:{1},avg_b:{2}'.format(avg_w1,avg_w2,avg_b))
    return (w1 - avg_w1 * lrw), (w2 - avg_w2 * lrw), (b - avg_b * lrb)


def gen_sample_data():
    w = random.randint(-60, -10) + random.random()
    b = random.randint(-45, 45) + random.random()

    print(w, b)

    num_sample = 100
    x1_list = []
    x2_list = []
    y_list = []
    for i in range(num_sample // 2):
        x1 = random.randint(0, 100) * random.random()
        x2 = w * x1 + b + random.random() * random.randint(100, 2000)
        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(1)
    for i in range(num_sample // 2):
        x1 = random.randint(0, 100) * random.random()
        x2 = w * x1 + b - random.random() * random.randint(100, 2000)
        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(0)
    return x1_list, x2_list, y_list


def get_help(w1, w2, b, x1):
    if w2 != 0:
        return -(b + w1 * x1) / w2
    return 9999999


def train(x1_list, x2_list, gt_y_list, batch_size, lrw, lrb, max_iter):
    w1 = 0
    w2 = 1
    b = 0
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x1_list), batch_size)
        batch_x1 = [x1_list[j] for j in batch_idxs]
        batch_x2 = [x2_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w1, w2, b = cal_step_gradient(batch_x1, batch_x2, batch_y, w1, w2, b, lrw, lrb)
        if (i % 1000 == 0):
            # print('w1:{0},w2:{1},b:{2}'.format(w1, w2, b))
            print('w:{0},b:{1}'.format(-w1 / w2, - b / w2))
            plt.scatter(x1_list, x2_list, c=y_list)
            xx = np.arange(0, 100, 1)
            yy = []
            for x in xx:
                yy.append(get_help(w1, w2, b, x))
            plt.plot(xx, yy)
            plt.show()
            # time.sleep(1)


x1_list, x2_list, y_list = gen_sample_data()

train(x1_list, x2_list, y_list, 100, 100, 100, 10000)
