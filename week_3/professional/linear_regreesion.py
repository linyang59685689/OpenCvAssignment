# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/10/26 下午8:32
# @Author: LiLinYang
# @File  : linear_regreesion.py


import numpy as np
import random
import time
import matplotlib.pyplot as plt


# linear regression

# 推测y
def inference(w, b, x):
    pred_y = w * x + b
    return pred_y


# 计算loss方法
def eval_loss(w, b, x_list, gt_y_list):
    sum_loss = 0
    for i in range(len(x_list)):
        sum_loss += (x_list[i] * w + b - gt_y_list[i]) ** 2
    return sum_loss / len(x_list)


# 单一样本带来的梯度
def gradient(x, pred_y, gt_y):
    dw = (pred_y - gt_y) * x
    db = (pred_y - gt_y)
    return dw, db


def cal_step_gradient(x_list, y_list, w, b, lrw,lrb):
    avg_w = 0
    avg_b = 0
    for i in range(len(x_list)):
        pred_y = inference(w, b, x_list[i])
        dw, db = gradient(x_list[i], pred_y, y_list[i])
        avg_w += dw
        avg_b += db
    avg_w /= len(x_list)
    avg_b /= len(x_list)

    return (w - avg_w * lrw), (b - avg_b * lrb)


def gen_sample_data():
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()*random.randint(-1,10)

    print(w, b)

    num_sample = 100
    x_list = []
    y_list = []
    for i in range(num_sample):
        x = random.randint(0, 100) * random.random()
        y = w * x + b+ random.random() * random.randint(-1, 20)

        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def train(x_list, gt_y_list, batch_size, lrw,lrb ,max_iter):
    w = 0
    b = 0
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lrw,lrb)
        if(i%10==0):
            # print('w:{0},b:{1}'.format(w, b))
            # print('loss is {}'.format(eval_loss(w, b, x_list, gt_y_list)))
            plt.scatter(x_list, gt_y_list)
            xx=[0,100]
            yy=[]
            yy.append(inference(w,b, xx[0]))
            yy.append(inference(w, b, xx[1]))
            plt.plot(xx,yy)
            plt.show()
            time.sleep(2)



x_list, y_list = gen_sample_data()
train(x_list, y_list, 100, 0.0001, 0.01,100)
