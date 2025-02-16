# -*- coding: utf-8 -*-
import numpy as np


def motion_axxb(hb2gs, hc2os):
    '''
    根据Hb2gs和Hc2os获取A和B,用于构建AX=XB
    :param hb2gs: list<array<4*4>> 机器臂姿态
    :param hc2os: list<array<4*4>> 相机外参
    :return: A,B:list<array<4*4>> AX=XB的参数
    '''
    a = []
    b = []
    n = len(hb2gs)
    k=0
    for i in range(n-1):
        for j in range(i,n):
            # print i,j,k
            # k=k+1
            a.append(np.dot(np.linalg.inv(hb2gs[j]),hb2gs[i]))
            b.append(np.dot(hc2os[j],np.linalg.inv(hc2os[i])))
    return a,b


def motion_axyb(hb2gs,hc2os):
    '''
    根据Hb2gs和Hc2os获取A和B,用于构建AX=XB
    :param hb2gs: list<array<4*4>> 机器臂姿态
    :param hc2os: list<array<4*4>> 相机外参
    :return: A,B:list<array<4*4>> AX=YB的参数
    '''
    a = []
    for i in range(len(hb2gs)):
        H = hc2os[i]
        a.append(np.linalg.inv(H))
    return hb2gs, a

