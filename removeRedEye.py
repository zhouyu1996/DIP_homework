#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@File    : removeRedEye.py
@Time    : 2019/10/26 11:55
@Author  : yu zhou
@content : 实现对人脸照片中的红眼去除
@Software: PyCharm
"""
import cv2
import numpy as np

# 填充模板空洞的函数
def fillHoles(mask):
    maskFloodfill = mask.copy()
    h, w = maskFloodfill.shape[:2]
    maskTemp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(maskFloodfill)
    return mask2 | mask

# 定义去红眼的函数，接受带有红眼的3通道图像，返回同样大小的图像
def removeRedEye(eye):
    eyeout = eye.copy()
    # 彩色图像分为rgb三个通道，opencv中imread得到的图像通道时bgr
    b = eye[:, :, 0]
    g = eye[:, :, 1]
    r = eye[:, :, 2]
    bg = cv2.add(b, g)
    # 去红眼模板
    mask = (r > 135) & (r > bg)
    # 类型转换
    mask = mask.astype(np.uint8) * 255
    # 使用flooddill和膨胀操作填充模板MASK中的空洞
    mask = fillHoles(mask)
    mask1 = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)
    # if mask1.shape== mask.shape:
    #     print("ds")
    # 利用模板进行去红眼操作，使用了（b+g）/2替代r通道中超过阈值的pixel
    mean = bg / 2
    #mask = mask.astype(np.bool)[:, :, np.newaxis]
    mean = mean[:, :, np.newaxis]

    mask_f = np.zeros(mask1.shape,np.bool)
    mask_f = mask_f[:, :, np.newaxis]
    mask_f[:,:,0] = mask1
    eyeout = np.where(mask_f, mean, eye)

    return eyeout

if __name__ == "__main__":
    # 加载图像
    img = cv2.imread("data/example1.jpg", cv2.IMREAD_COLOR)
    #cv2.imshow("source",img)
    # opencv的imread方法不会判断读取的图像是否成功
    if img is None:
        print("加载图片失败")
    # 备份图像一遍输出对比
    imgCopy = img.copy()
    # 加载opencv提供的级联眼睛检测器
    # haarcascade_eye.xml是opencv提供的方法，可以根据参数检测眼睛区域
    eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    # 完成眼部区域的检测
    eyes = eyesCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))
    # 对每一个检测到的眼睛区域进行去红眼操作
    for (x, y, w, h) in eyes:
        # # 显示检测出的眼睛区域
        # tmp = imgCopy
        # cv2.rectangle(tmp,(x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("eye",tmp)

        # 截取图像中的眼睛区域
        eye = img[y:y + h, x:x + w]
        # 去除截取部分的红眼
        eyeOut = removeRedEye(eye)
        # 将去除红眼之后的眼睛区域替换掉原来的红眼区域
        imgCopy[y:y + h, x:x + w, :] = eyeOut
        #
        # cv2.rectangle(tmp,(x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.waitKey()
    # 对比展示去红眼的结果
    cv2.imshow('Red Eyes', img)
    cv2.imshow('Red Eyes Removed', imgCopy)
    cv2.waitKey()
    cv2.destroyAllWindows()
