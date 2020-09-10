#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@File    : faceBeauty.py
@Time    : 2019/10/26 13:02
@Author  : yu zhou
@content : 对人脸进行美化
@Software: PyCharm
"""

import cv2
import numpy as np

def face_beauty(img):
    a=1

if __name__ =="__main__":

    img =cv2.imread('data/beauty.jpg', cv2.IMREAD_COLOR)
    # opencv的imread方法不会判断读取的图像是否成功
    if img is None:
        print("加载图片失败")
    # 备份图像一遍输出对比
    imgCopy = img.copy()
    # 加载opencv提供的级联人脸检测器
    # haarcascade_frontalface_alt2.xml是opencv提供的方法，可以根据参数检测人脸区域
    face = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # 完成人脸区域的检测
    faces = face.detectMultiScale(img, scaleFactor=1.3, minNeighbors=2, minSize=(50, 50))
    # 对每一个检测到的眼睛区域进行去红眼操作
    for (x, y, w, h) in faces:
        # 显示检测出的人脸区域
        # tmp = imgCopy
        # cv2.rectangle(tmp,(x, y), (x + w, y + h), ( 0, 0, 255), 2)
        # cv2.imshow("face",tmp)
        # cv2.waitKey()
        # 截取图像中的眼睛区域
        face = img[y:y + h, x:x + w]
        # 去除截取部分的红眼
        faceout = face_beauty(face)
        # 将去除红眼之后的眼睛区域替换掉原来的红眼区域
        imgCopy[y:y + h, x:x + w, :] = faceout
        # cv2.rectangle(tmp,(x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.waitKey()
    # 对比展示
    cv2.imshow('Red Eyes', img)
    cv2.imshow('Red Eyes Removed', imgCopy)
    cv2.waitKey()
    cv2.destroyAllWindows()


