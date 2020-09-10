#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@File    : imgBeauty.py
@Time    : 2019/10/26 13:02
@Author  : yu zhou
@content : 对人脸进行美化
@Software: PyCharm
"""

import cv2
import numpy as np


def whitening(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            (b, g, r) = img[i, j]
            bb = int(b * 1.2) + 10
            gg = int(g * 1.2) + 10
            if bb > 255:
                bb = 255
            if gg > 255:
                gg = 255
            dst[i, j] = (bb, gg, r)
    return dst

def buffing(img):

    # src：输入图像
    # d：过滤时周围每个像素领域的直径
    # sigmaColor：在color space中过滤sigma。参数越大，临近像素将会在越远的地方mix。
    # sigmaSpace：在coordinate space中过滤sigma。参数越大，那些颜色足够相近的的颜色的影响越大。
    dst = cv2.bilateralFilter(img, 15, 35, 35)
    dst = cv2.bilateralFilter(dst, 30, 50, 20)
    return dst

# 本来计划实现只对人脸区域的美白函数
# 但是由于cv2中的人脸检测器给出的是矩形，且美白前后头像差别很大 因此还原后效果很差
# 要实现这个可能可能需要基于人脸轮廓进行区域检测，且美白算法要比较自然
# 留着此处将来有机会再进行优化
def face_whitening(img):
    facesCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # 完成人脸区域的检测
    faces = facesCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=2, minSize=(100, 100))
    for (x, y, w, h) in faces:
        # # 显示检测出的人脸区域
        # tmp = img
        # cv2.rectangle(tmp,(x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("face",tmp)

        # 截取图像中的人脸区域
        face = img[y:y + h, x:x + w]
        # 对人脸进行美白
        faceout = whitening(face)
        # 将去除红眼之后的眼睛区域替换掉原来的红眼区域
        img[y:y + h, x:x + w, :] = faceout
        cv2.imshow('1',face)
        cv2.imshow("2",faceout)
        cv2.waitKey()
    return img

if __name__ =="__main__":
    img =cv2.imread('data/beauty.jpg', cv2.IMREAD_COLOR)
    # opencv的imread方法不会判断读取的图像是否成功
    if img is None:
        print("加载图片失败")
    # 备份图像一遍输出对比
    imgCopy = img.copy()
    # 对比展示
    cv2.imshow('sourse',img)
    #cv2.imshow('buffing', face_whitening(imgCopy))
    #cv2.imshow('buffing',buffing(imgCopy))
    #cv2.imshow('buffing+whitening', whitening(buffing(imgCopy)))
    # cv2.waitKey()
    #cv2.imshow('whitening', whitening(imgCopy))
    cv2.imshow('whitening+buffing', buffing(whitening(imgCopy)))
    cv2.waitKey()
    cv2.destroyAllWindows()