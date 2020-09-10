#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/30 16:22
# @Author  : yu zhou
# @Site    : 
# @File    : carNumberDetect.py
# @content : 对图像中的多个车牌进行定位
# @Software: PyCharm
import cv2
import numpy as np

def verify_scale(rotate_rect):
   error = 0.4
   aspect = 4 #4.7272
   min_area = 10*(10*aspect)
   max_area = 150*(150*aspect)
   min_aspect = aspect*(1-error)
   max_aspect = aspect*(1+error)
   theta = 30

   # 宽或高为0，不满足矩形直接返回False
   if rotate_rect[1][0]==0 or rotate_rect[1][1]==0:
       return False

   r = rotate_rect[1][0]/rotate_rect[1][1]
   r = max(r,1/r)
   area = rotate_rect[1][0]*rotate_rect[1][1]
   if area>min_area and area<max_area and r>min_aspect and r<max_aspect:
       # 矩形的倾斜角度在不超过theta
       if ((rotate_rect[1][0] < rotate_rect[1][1] and rotate_rect[2] >= -90 and rotate_rect[2] < -(90 - theta)) or
               (rotate_rect[1][1] < rotate_rect[1][0] and rotate_rect[2] > -theta and rotate_rect[2] <= 0)):
           return True
   return False

def pre_process(orig_img):
    # 获取灰度图
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray_img', gray_img)
    # 均值模糊,柔化一些小的噪声点
    blur_img = cv2.blur(gray_img, (3, 3))
    # cv2.imshow('blur', blur_img)

    # sobel获取垂直边缘：因为车牌垂直边缘比较多
    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)
    #cv2.imshow('sobel', sobel_img)

    # 转换到hsv空间进行颜色定位
    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 黄色色调区间[26，34],蓝色色调区间:[100,124]
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')

    # 将颜色定位后的照片和sobel检测到的区域相乘
    mix_img = np.multiply(sobel_img, blue_img)
    #cv2.imshow('mix', mix_img)

    mix_img = mix_img.astype(np.uint8)
    # 二值化：最大类间方差法
    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('binary',binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('close', close_img)

    return close_img

# 给候选车牌区域做漫水填充算法，一方面补全上一步求轮廓可能存在轮廓歪曲的问题，
# 另一方面也可以将非车牌区排除掉
def verify_color(rotate_rect,src_image):
    img_h,img_w = src_image.shape[:2]
    mask = np.zeros(shape=[img_h+2,img_w+2],dtype=np.uint8)
    connectivity = 8 #种子点上下左右4邻域与种子颜色值在[loDiff,upDiff]的被涂成new_value，也可设置8邻域
    loDiff,upDiff = 30,30
    new_value = 255
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE  #考虑当前像素与种子象素之间的差，不设置的话则和邻域像素比较
    flags |= new_value << 8
    flags |= cv2.FLOODFILL_MASK_ONLY #设置这个标识符则不会去填充改变原始图像，而是去填充掩模图像（mask）

    rand_seed_num = 50 #生成多个随机种子
    valid_seed_num = 10 #从rand_seed_num中随机挑选valid_seed_num个有效种子
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)
    box_points_x = [n[0] for n in box_points]
    box_points_x.sort(reverse=False)
    adjust_x = int((box_points_x[2]-box_points_x[1])*adjust_param)
    col_range = [box_points_x[1]+adjust_x,box_points_x[2]-adjust_x]
    box_points_y = [n[1] for n in box_points]
    box_points_y.sort(reverse=False)
    adjust_y = int((box_points_y[2]-box_points_y[1])*adjust_param)
    row_range = [box_points_y[1]+adjust_y, box_points_y[2]-adjust_y]
    # 如果以上方法种子点在水平或垂直方向可移动的范围很小，则采用旋转矩阵对角线来设置随机种子点
    if (col_range[1]-col_range[0])/(box_points_x[3]-box_points_x[0])<0.4\
        or (row_range[1]-row_range[0])/(box_points_y[3]-box_points_y[0])<0.4:
        points_row = []
        points_col = []
        for i in range(2):
            pt1,pt2 = box_points[i],box_points[i+2]
            x_adjust,y_adjust = int(adjust_param*(abs(pt1[0]-pt2[0]))),int(adjust_param*(abs(pt1[1]-pt2[1])))
            if (pt1[0] <= pt2[0]):
                pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
            else:
                pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
            if (pt1[1] <= pt2[1]):
                pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
            else:
                pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
            temp_list_x = [int(x) for x in np.linspace(pt1[0],pt2[0],int(rand_seed_num /2))]
            temp_list_y = [int(y) for y in np.linspace(pt1[1],pt2[1],int(rand_seed_num /2))]
            points_col.extend(temp_list_x)
            points_row.extend(temp_list_y)
    else:
        points_row = np.random.randint(row_range[0],row_range[1],size=rand_seed_num)
        points_col = np.linspace(col_range[0],col_range[1],num=rand_seed_num).astype(np.int)

    points_row = np.array(points_row)
    points_col = np.array(points_col)
    hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h,s,v = hsv_img[:,:,0],hsv_img[:,:,1],hsv_img[:,:,2]
    # 将随机生成的多个种子依次做漫水填充,理想情况是整个车牌被填充
    flood_img = src_image.copy()
    seed_cnt = 0
    for i in range(rand_seed_num):
        rand_index = np.random.choice(rand_seed_num,1,replace=False)
        row,col = points_row[rand_index],points_col[rand_index]
        # 限制随机种子必须是车牌背景色
        if (((h[row,col]>26)&(h[row,col]<34))|((h[row,col]>100)&(h[row,col]<124)))&(s[row,col]>70)&(v[row,col]>70):
            cv2.floodFill(src_image, mask, (col,row), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
            cv2.circle(flood_img,center=(col,row),radius=2,color=(0,0,255),thickness=2)
            seed_cnt += 1
            if seed_cnt >= valid_seed_num:
                break
    #======================调试用======================#
    # show_seed = np.random.uniform(1,100,1).astype(np.uint16)
    # cv2.imshow('floodfill'+str(show_seed),flood_img)
    # cv2.imshow('flood_mask'+str(show_seed),mask)
    #======================调试用======================#
    # 获取掩模上被填充点的像素点，并求点集的最小外接矩形
    mask_points = []
    for row in range(1,img_h+1):
        for col in range(1,img_w+1):
            if mask[row,col] != 0:
                mask_points.append((col-1,row-1))
    mask_rotateRect = cv2.minAreaRect(np.array(mask_points))
    if verify_scale(mask_rotateRect):
        return True, mask_rotateRect
    else:
        return False, mask_rotateRect

# 车牌定位
def locate_carPlate(orig_img,pred_image):
    carPlate_list = []
    temp1_orig_img = orig_img.copy()
    temp2_orig_img = orig_img.copy()
    # findContours在3.4.3.18的版本之前输出三个
    # 新的版本输出两个
    t,contours,heriachy = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
        # 获取轮廓最小外接矩形，返回值rotate_rect
        rotate_rect = cv2.minAreaRect(contour)
        # 根据矩形面积大小和长宽比判断是否是车牌
        if verify_scale(rotate_rect):
            ret,rotate_rect2 = verify_color(rotate_rect,temp1_orig_img)
            if ret == False:
                continue
            box = cv2.boxPoints(rotate_rect2)
            for k in range(4):
                n1,n2 = k%4,(k+1)%4
                cv2.line(temp2_orig_img,(box[n1][0],box[n1][1]),(box[n2][0],box[n2][1]),(255,0,0),2)
    cv2.imshow('contour', temp2_orig_img)
    cv2.waitKey()
    return temp1_orig_img

if __name__ == '__main__':
    img = cv2.imread('data/car.jpg',cv2.IMREAD_COLOR)
    # 预处理
    pred_img = pre_process(img)
    # 车牌定位
    car_plate_list = locate_carPlate(img,pred_img)
    cv2.imshow('car_plate',car_plate_list)
    cv2.waitKey()
    cv2.destroyAllWindows()