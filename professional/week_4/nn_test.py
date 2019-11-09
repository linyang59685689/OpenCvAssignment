# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/9 下午9:45
# @Author: LiLinYang
# @File  : nn_test.py

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

np.random.seed(3)
X, y = sklearn.datasets.make_circles(200, noise=0.1, factor=0.4)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2

lr = 0.01
reg_lambda = 0.01


def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1,a1, z2,exp_scores,probs = forward_cal(W1, W2, b1, b2)
    corect_logprobs=-np.log(probs[range(num_examples),y])
    data_loss=np.sum(corect_logprobs)
    return 1/num_examples*data_loss


def predict(model,x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1,a1, z2,exp_scores,probs = forward_cal(W1, W2, b1, b2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs,axis=1)


def build_model(nn_hdim,num_passes=30000,print_loss=False):
    np.random.seed(2)
    W1=np.random.randn(nn_input_dim,nn_hdim)/np.sqrt(nn_input_dim)
    b1=np.zeros((1,nn_hdim))
    W2=np.random.randn(nn_hdim,nn_output_dim)/np.sqrt(nn_hdim)
    b2=np.zeros((1,nn_output_dim))

    model={}

    for i in range(0,num_passes):
        z1,a1, z2,exp_scores,probs = forward_cal(W1, W2, b1, b2)

        delta3=probs
        delta3[range(num_examples),y]-=1

        dW2=(a1.T).dot(delta3)
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(W2.T)*(1-np.power(a1,2))
        dW1=np.dot(X.T,delta2)
        db1=np.sum(delta2,axis=0)

        W1+=-lr*dW1
        b1+=-lr*db1
        W2+=-lr*dW2
        b2+=-lr*db2

        model={'W1':W1,'b1':b1,'W2':W2,'b2':b2}

        if print_loss and i%1000 ==0:
            print("Loss"
                  "loss after iteration %i: %f"%(i,calculate_loss(model)))
    return model


def forward_cal(W1, W2, b1, b2):
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return z1,a1, z2,exp_scores,probs


model1=build_model(3,print_loss=True)