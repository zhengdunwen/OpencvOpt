#-*- coding: UTF-8 -*-

'''
Author: Steve Wang
Time: 2017/12/8 10:00
Environment: Python 3.6.2 |Anaconda 4.3.30 custom (64-bit) Opencv 3.3
'''
import cv2
from math import *
import numpy as np
import os
import time


def get_image(path):
    #获取图片
    img=cv2.imread(path)
    print(path)
    blurred = cv2.medianBlur(img, 9)
    #dilates = cv2.dilate(blurred, None, iterations=2)
    #dilates = cv2.erode(blurred, None, iterations=6)
    img_ = cv2.medianBlur(blurred, 9)


    gray=cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

    return img, gray

def Gaussian_Blur(gray):
    # 高斯去噪
    blurred = cv2.GaussianBlur(gray, (5,5),0)

    return blurred

def Sobel_gradient(blurred):
    # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient

def Thresh_and_blur(gradient):
    #blurred = cv2.GaussianBlur(gradient, (35, 35),0)
    (_, thresh) = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)
    return thresh

def Thresh_and_blur_cut(gradient):
    blurred = cv2.GaussianBlur(gradient, (3, 3),0)
    (_, thresh) = cv2.threshold(gradient, 220, 255, cv2.THRESH_BINARY)
    return thresh

def image_morphology(thresh):
    # 建立一个椭圆核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # 执行图像形态学, 细节直接查文档，很简单
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, None, iterations=4)
    #closed = cv2.erode(closed, None, iterations=4)
    return closed



def findcnts_and_box_point(closed):
    # 这里opencv3返回的是三个参数
    Imagesss = closed.copy()
    (image, cnts, _hierarchy) = cv2.findContours(Imagesss,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_L1)

    #按照大小进行排序
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    print(np.size(cnts))  # 得到该图中总的轮廓数量
    print(np.size(c))  # 得到该图中总的轮廓数量
    box = []
    # compute the rotated bounding box of the largest contour
    for i in c:
        rect = cv2.minAreaRect(i)
        #print('宽:' + str(rect[1][0]) + '    高:' + str(rect[1][1]))
        if rect[1][0] > 30 or rect[1][1] > 30:
            box.append( np.int0(cv2.boxPoints(rect)))

    '''
    box_sort =sorted(box, key=lambda xa: xa[1][0] + xa[1][1]//100 * 600,  reverse=False)  # sort by age  [x,y]

    nums =1

    for n in box_sort:
        n[np.lexsort(n.T)]
        print(str(nums) + ':' + str(n[0][0] + n[0][1]//100*2000))
        nums =nums +1
    '''

    return box


def Cut_File(original_img):
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurs_ = Thresh_and_blur_cut(gray)


    blurs_ = 255 - blurs_
    # blurs_ = cv2.dilate(blurs_, None, iterations=2)

    # 找到轮廓
    _, contours, hierarchy = cv2.findContours(blurs_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)#按照大小排序
    box = []
    cut_num = 0
    for i in contours_sorted:
        rect = cv2.minAreaRect(i)
        #过滤小的轮廓 最多只是取2个区域的数据
        print('宽:' + str(rect[1][0]) + '    高:' + str(rect[1][1]))
        if cut_num <2 and (rect[1][0] > 20 or rect[1][1] > 20):
            box.append(np.int0(cv2.boxPoints(rect)))
            cut_num = cut_num + 1

    # 绘制轮廓
    #cv2.drawContours(original_img, box, -1, (0, 255, 0), 1)
    #cv2.imshow(str(time.time()), original_img)


    copy_image = []
    for i in box:
        Xs = [n[0] for n in i]
        Ys = [n[1] for n in i]
        x1 = min(Xs)
        x2 = max(Xs)

        y1 = min(Ys)
        y2 = max(Ys)

        if x1 < 0:
            x1 = 0
        if x2 < 0:
            x2 = 0
        if y1 < 0:
            y1 = 0
        if y2 < 0:
            y2 = 0

        h = y2 - y1
        w = x2 - x1
        copy_image.append(original_img[y1:y2, x1:x2])
        #cv2.rectangle(proimage, (x1, y1), (x1 + w, y1 + h), (153, 153, 0), 1)
        #cv2.imshow('copy_image' + str(i), original_img[y1:y2, x1:x2])

    #cv2.imshow('blurs_', blurs_)
    #cv2.imshow('original_img', original_img)
    #cv2.imshow('proimage', proimage)
    return copy_image



def drawcnts_and_cut(original_img, box):
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    # draw a bounding box arounded the detected barcode and display the image
    h, w = original_img.shape[:2]  # 获取图像的高和宽

    xBox = []
    for x in box:
        Xs = [i[0] for i in x]
        Ys = [i[1] for i in x]
        x1 = min(Xs)
        x2 = max(Xs)

        y1 = min(Ys)
        y2 = max(Ys)

        if x1 < 0:
            x1 = 0
        if x2 < 0:
            x2 = 0
        if y1 < 0:
            y1 = 0
        if y2 < 0:
            y2 = 0

        #hight = y2 - y1
        #width = x2 - x1

        #crop_img .append(original_img[y1:y1 + hight, x1:x1 + width])
        arr = [[x1,y2],[x1,y1],[x2,y1],[x2,y2]]
        xBox.append(arr)

    box_sort = sorted(xBox, key=lambda xa: xa[1][0] + xa[1][1] // 100 * w, reverse=False)  # sort by age  [x,y]

    numx = 1
    crop_img_dist ={}

    for nx in box_sort:
        x1 = nx[1][0]
        x2 = nx[3][0]

        y1 = nx[1][1]
        y2 = nx[3][1]

        hight = y2 - y1
        width = x2 - x1

        if numx <= 24:
            crop_img_dist[numx] = Cut_File(original_img[y1:y1 + hight, x1:x1 + width])
            #cv2.imwrite('D:\\tensorflow_code\\DNA_TEST\\data\\' + str(numx) + '.jpg', original_img[y1:y1 + hight, x1:x1 + width])
        numx = numx + 1


        #cv2.rectangle(original_img, (x1, y1), (x2, y2), (55, 255, 155), 2)
       # cv2.rectangle(original_img, (x1, y1), (x2,y2), (55, 255, 155), 2)


    draw_img = cv2.drawContours(original_img.copy(), box, -1, (0, 0, 255), 1)


    return draw_img,crop_img_dist


def save_cut_files(img,crop_img_dist,save_path,img_file_name_):
    filename =  os.path.splitext(img_file_name_)
    for boxKey in crop_img_dist:
        boxList = crop_img_dist[boxKey]
        num = 0
        for n in boxList:
            img_cut = n

            height, width = img_cut.shape[:2]
            strs = str(boxKey)

            for c in range(0, 1):
                degree = c * 10
                heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
                widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

                matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

                matRotation[0, 2] += (widthNew - width) / 2
                matRotation[1, 2] += (heightNew - height) / 2
                imgRotation = cv2.warpAffine(img_cut, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path + '\\'+filename[0] +'+'+ strs +'+'+ str(num) +'+' +str(c * 10) + '.jpg', imgRotation)

                # cv2.imshow(strs, imgRotation)
            num = num + 1



def file_analysis(img_path,img_file_name_,save_path):

    #img_path = r'D:\tensorflow_code\DNA_TEST\data\30240_K.jpg'
    #img_file_name_= '30240_K.jpg'
    #save_path = r'D:\tensorflow_code\DNA_TEST\data\CutFiles'
    original_img, gray = get_image(img_path+img_file_name_)
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    box = findcnts_and_box_point(closed)
    draw_img, crop_img_dist = drawcnts_and_cut(original_img,box)



    # 暴力一点，把它们都显示出来看看
    '''
    cv2.imshow('original_img', original_img)
    cv2.imshow('blurred', blurred)
    cv2.imshow('gradX', gradX)
    cv2.imshow('gradY', gradY)
    '''
    cv2.imshow('final', gradient)
    cv2.imshow('thresh', thresh)
    cv2.imshow('closed', closed)
    cv2.imshow('draw_img', draw_img)

    save_cut_files(original_img,crop_img_dist,save_path,img_file_name_)


def get_file_list(img_path,save_path):
    if os.path.isfile(img_path):
        return

    group_info = os.walk(img_path)

    for _,_,files in group_info:
        for filename in files:
            fn = os.path.splitext(filename)
            if fn[1] == '.jpeg':
                file_analysis(img_path,filename,save_path)
        return
    return


file_analysis('D:\\tensorflow_code\DNA_TEST\\DNA_862\\JPEG-Cut\\',
              '3110c2e4-beb1-435c-b26f-2663a5dac4dc.jpeg',
              'D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\3110c2e4-beb1-435c-b26f-2663a5dac4dc\\')



#get_file_list('D:\\tensorflow_code\\DNA_TEST\\DNA_862\\JPEG-Cut\\','D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\A\\')
cv2.waitKey(20171219)

