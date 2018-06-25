# -*- coding: UTF-8 -*-

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
from PIL import Image


def get_image(path):
    # 获取图片
    img = cv2.imread(path)
    blurred = cv2.medianBlur(img, 9)
    # dilates = cv2.dilate(blurred, None, iterations=2)
    # dilates = cv2.erode(blurred, None, iterations=6)
    img_ = cv2.medianBlur(blurred, 9)

    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    return img, gray


def Gaussian_Blur(gray):
    # 高斯去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred


def Sobel_gradient(blurred):
    # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient


def Thresh_and_blur(gradient):
    # blurred = cv2.GaussianBlur(gradient, (35, 35),0)
    (_, thresh) = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)
    return thresh


def Thresh_and_blur_cut(gradient):
    blurred = cv2.GaussianBlur(gradient, (3, 3), 0)
    (_, thresh) = cv2.threshold(gradient, 220, 255, cv2.THRESH_BINARY)
    return thresh


def image_morphology(thresh):
    # 建立一个椭圆核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # 执行图像形态学, 细节直接查文档，很简单
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    closed = cv2.dilate(closed, kernel, iterations=10)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    # closed = cv2.erode(closed, kernel, iterations=2)
    return closed


def findcnts_and_box_point(closed):
    # 这里opencv3返回的是三个参数
    Imagesss = closed.copy()
    (image, cnts, _hierarchy) = cv2.findContours(Imagesss,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_TC89_L1)

    # 按照大小进行排序
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    # print(np.size(cnts))  # 得到该图中总的轮廓数量
    # print(np.size(c))  # 得到该图中总的轮廓数量
    box = []
    # compute the rotated bounding box of the largest contour
    for i in c:
        rect = cv2.minAreaRect(i)
        # print('宽:' + str(rect[1][0]) + '    高:' + str(rect[1][1]))
        if rect[1][0] > 25 and rect[1][1] > 25:
            box.append(np.int0(cv2.boxPoints(rect)))

    '''
    box_sort =sorted(box, key=lambda xa: xa[1][0] + xa[1][1]//100 * 600,  reverse=False)  # sort by age  [x,y]

    nums =1

    for n in box_sort:
        n[np.lexsort(n.T)]
        print(str(nums) + ':' + str(n[0][0] + n[0][1]//100*2000))
        nums =nums +1
    '''

    return box


# 把2个DNA分离开
def Cut_File(DNANum, original_img):
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurs_ = Thresh_and_blur_cut(gray)

    blurs_ = 255 - blurs_
    # blurs_ = cv2.dilate(blurs_, None, iterations=2)

    # 找到轮廓
    _, contours, hierarchy = cv2.findContours(blurs_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)  # 按照大小排序
    box = []
    cut_num = 0

    # 24号染色体只能有一只
    if DNANum == 24:
        maxNum = 1
    else:
        maxNum = 2

    for i in contours_sorted:
        rect = cv2.minAreaRect(i)
        # 过滤小的轮廓 最多只是取2个区域的数据

        if cut_num < maxNum and (rect[1][0] > 14 and rect[1][1] > 14):
            box.append(np.int0(cv2.boxPoints(rect)))
            cut_num = cut_num + 1
            # print(str(DNANum) + '  ' + str(cut_num)  + '  宽:' + str(int(rect[1][0])) + '    高:' + str(int(rect[1][1])))

    # 绘制轮廓
    # cv2.drawContours(original_img, box, -1, (0, 255, 0), 1)
    # cv2.imshow(str(time.time()), original_img)

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

        copy_image.append(ReCheck_Cut_File(original_img[y1:y2, x1:x2]))
        # copy_image.append(original_img[y1:y2, x1:x2])
        # cv2.rectangle(proimage, (x1, y1), (x1 + w, y1 + h), (153, 153, 0), 1)
        # cv2.imshow('copy_image' + str(i), original_img[y1:y2, x1:x2])

    # cv2.imshow('blurs_', blurs_)
    # cv2.imshow('original_img', original_img)
    # cv2.imshow('proimage', proimage)

    return copy_image


# 对剪切的文件进行二次检查
def ReCheck_Cut_File(original_img):
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurs_ = Thresh_and_blur_cut(gray)

    blurs_ = 255 - blurs_
    # blurs_ = cv2.dilate(blurs_, None, iterations=2)

    # 找到轮廓
    _, contours, hierarchy = cv2.findContours(blurs_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)  # 按照大小排序
    box = []
    cut_num = 0

    maxNum = 1

    for i in contours_sorted:
        rect = cv2.minAreaRect(i)
        # 过滤小的轮廓 最多只是取2个区域的数据

        if cut_num < maxNum and (rect[1][0] > 14 and rect[1][1] > 14):
            box.append(np.int0(cv2.boxPoints(rect)))
            cut_num = cut_num + 1
            # print(str(cut_num) + '  宽:' + str(int(rect[1][0])) + '    高:' + str(int(rect[1][1])))

    # 绘制轮廓
    # cv2.drawContours(original_img, box, -1, (0, 255, 0), 1)
    # cv2.imshow(str(time.time()), original_img)
    if cut_num <= 0:
        print("no box")
        return original_img

    # ROI的处理
    dst_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    dst_rect = np.array([box[0]], np.int32)
    cv2.fillPoly(dst_mask, dst_rect, 255)
    dst_mask_f = 255 - dst_mask

    # 取反向
    image_f = cv2.add(np.zeros(np.shape(original_img), dtype=np.uint8) + 255, original_img, mask=dst_mask_f)
    image = cv2.add(original_img, image_f)

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
        copy_image = image[y1:y2, x1:x2]
        # cv2.rectangle(proimage, (x1, y1), (x1 + w, y1 + h), (153, 153, 0), 1)
        # cv2.imshow('copy_image' + str(i), image[y1:y2, x1:x2])

    # cv2.imshow('blurs_', blurs_)
    # cv2.imshow('original_img', original_img)
    # cv2.imshow('proimage', proimage)
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

        # hight = y2 - y1
        # width = x2 - x1

        # crop_img .append(original_img[y1:y1 + hight, x1:x1 + width])
        arr = [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
        xBox.append(arr)

    # 画面划分成4行7列的格式，使用中心点的坐标进行区域的运算
    Hight_even = h // 4
    Weight_even = w // 7
    box_sort = sorted(xBox, key=lambda xa: (xa[0][0] + (xa[2][0] - xa[0][0]) // 2) // Weight_even + (
                xa[1][1] + (xa[0][1] - xa[1][1]) // 2) // Hight_even * 7, reverse=False)  # sort by age  [x,y]

    numx = 1
    crop_img_dist = {}

    for nx in box_sort:
        x1 = nx[1][0]
        x2 = nx[3][0]

        y1 = nx[1][1]
        y2 = nx[3][1]

        hight = y2 - y1
        width = x2 - x1

        if numx <= 24:
            crop_img_dist[numx] = Cut_File(numx, original_img[y1:y1 + hight, x1:x1 + width])
            # cv2.imshow(str(numx),original_img[y1:y1 + hight, x1:x1 + width])
            # print(str(numx))
            # print(str(numx) + '  宽:' + str(int(width)) + '    高:' + str(int(hight)))
            # cv2.imwrite('D:\\tensorflow_code\\DNA_TEST\\data\\' + str(numx) + '.jpg', original_img[y1:y1 + hight, x1:x1 + width])
        numx = numx + 1

        # cv2.rectangle(original_img, (x1, y1), (x2, y2), (55, 255, 155), 2)
    # cv2.rectangle(original_img, (x1, y1), (x2,y2), (55, 255, 155), 2)

    draw_img = cv2.drawContours(original_img.copy(), box, -1, (0, 0, 255), 1)

    return draw_img, crop_img_dist


def save_cut_files(img, crop_img_dist, save_path, img_file_name_):
    filename = os.path.splitext(img_file_name_)
    filenum = 0
    for boxKey in crop_img_dist:
        boxList = crop_img_dist[boxKey]
        num = 0
        for n in boxList:
            img_cut = n

            height, width = img_cut.shape[:2]
            strs = str(boxKey)

            Cross_num = 1  # 转换角度
            for c in range(0, Cross_num):
                degree = c * 360 // Cross_num
                heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
                widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

                matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

                matRotation[0, 2] += (widthNew - width) / 2
                matRotation[1, 2] += (heightNew - height) / 2
                imgRotation = cv2.warpAffine(img_cut, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path + '\\' + filename[0] + '+' + strs + '+' + str(num) + '+' + str(c * 10) + '.jpg',
                            imgRotation)

                # cv2.imshow(strs, imgRotation)
                filenum = filenum + 1
            num = num + 1
    if filenum != 23 * 2 * Cross_num:
        print('list num  ' + str(filenum) + '  Filname:  ' + img_file_name_)


def file_analysis(img_path, img_file_name_, save_path):
    # img_path = r'D:\tensorflow_code\DNA_TEST\data\30240_K.jpg'
    # img_file_name_= '30240_K.jpg'
    # save_path = r'D:\tensorflow_code\DNA_TEST\data\CutFiles'
    original_img, gray = get_image(img_path + img_file_name_)
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    box = findcnts_and_box_point(closed)
    draw_img, crop_img_dist = drawcnts_and_cut(original_img, box)

    # 暴力一点，把它们都显示出来看看
    '''
    cv2.imshow('original_img', original_img)
    cv2.imshow('blurred', blurred)
    cv2.imshow('gradX', gradX)
    cv2.imshow('gradY', gradY)
    cv2.imshow('final', gradient)
    cv2.imshow('thresh', thresh)

    cv2.imshow('closed', closed)
    cv2.imshow('draw_img', draw_img)
    '''

    save_cut_files(original_img, crop_img_dist, save_path, img_file_name_)


def get_file_list(img_path, save_path):
    if os.path.isfile(img_path):
        return

    group_info = os.walk(img_path)

    for _, _, files in group_info:
        for filename in files:
            fn = os.path.splitext(filename)
            if fn[1] == '.jpeg':
                print(filename)
                file_analysis(img_path, filename, save_path)
        return
    return


def get_file_list_dir(img_path, save_path):
    if os.path.isfile(img_path):
        return
    fs = os.listdir(img_path)
    bdst = 0
    for f1 in fs:
        tmp_path = os.path.join(img_path, f1)
        if not os.path.isdir(tmp_path):
            if os.path.splitext(tmp_path)[1] == '.jpeg' or os.path.splitext(tmp_path)[1] == '.jpg':
                img = Image.open(tmp_path)
                if (img.size == (1232, 912) or tmp_path.find('Karyotype') > 0) and bdst == 0:
                    print(f1)
                    file_analysis(img_path + "\\", f1, save_path)
                    bdst = 1
        else:
            print('文件夹：', tmp_path)
            get_file_list_dir(tmp_path, save_path)
    return


if __name__ == '__main__':
    '''
    fo = open("C:\\Users\\zdw\\Desktop\\123.txt", "r+")
    print ("文件名为: ", fo.name)
    line = fo.readline()
    while len(line) > 0:
        line = line[-42:-1]
        file_analysis('D:\\tensorflow_code\DNA_TEST\\DNA_862\\JPEG-Cut\\',
                      line,
                      'D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\3110c2e4-beb1-435c-b26f-2663a5dac4dc\\')
        line = fo.readline()


    file_analysis('D:\\tensorflow_code\DNA_TEST\\DNA_862\\JPEG-Cut\\',
                  'dd151bd6-2071-4d4c-b165-e1f79d75a50d.jpeg',
                  'D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\3110c2e4-beb1-435c-b26f-2663a5dac4dc\\')




    '''
    # get_file_list('D:\\tensorflow_code\\DNA_TEST\\DNA_862\\JPEG-Cut\\','D:\\tensorflow_code\\DNA_TEST\\DNA_862\\JPEG_Cut_Fenjie_3\\')
    # img = cv2.imread(r'D:\tensorflow_code\DNA_TEST\100_R_Test\Cut_100_1\0c3c290f-2140-4585-ba39-45fe29e9fae5+1+0+0.jpg')
    # ReCheck_Cut_File(img)

    get_file_list_dir('D:\\tensorflow_code\\DNA_TEST\\DNA_862\\862_File_remove\\', \
                      'D:\\tensorflow_code\\DNA_TEST\\DNA_862\\JPEG_Cut_Fenjie_all\\')
    cv2.waitKey(20171219)

