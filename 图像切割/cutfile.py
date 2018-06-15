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
    blurred = cv2.GaussianBlur(gradient, (35, 35),0)
    (_, thresh) = cv2.threshold(gradient, 220, 255, cv2.THRESH_BINARY)
    return thresh




def image_morphology(thresh):
    # 建立一个椭圆核函数
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # 执行图像形态学, 细节直接查文档，很简单
    #closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed = cv2.dilate(thresh, kernel, iterations=18)

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
        box.append(np.int0(cv2.boxPoints(rect)))
        break

    '''
    box_sort =sorted(box, key=lambda xa: xa[1][0] + xa[1][1]//100 * 600,  reverse=False)  # sort by age  [x,y]

    nums =1

    for n in box_sort:
        n[np.lexsort(n.T)]
        print(str(nums) + ':' + str(n[0][0] + n[0][1]//100*2000))
        nums =nums +1
    '''

    return box


def Rotation(img_cut,degree,width,height):
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img_cut, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return  imgRotation


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

    box_sort = sorted(xBox, key=lambda xa: (xa[3][0] - xa[1][0])*(xa[3][1]-xa[1][1]), reverse=True)  # sort by age  [x,y]

    numx = 1
    crop_img_dist = {}

    for nx in box_sort:
        x1 = nx[1][0]
        x2 = nx[3][0]

        y1 = nx[1][1]
        y2 = nx[3][1]

        hight = y2 - y1
        width = x2 - x1

        if numx <= 1:
            crop_img_dist[numx] = original_img[y1:y1 + hight, x1:x1 + width]
            #cv2.imshow(str(numx),original_img[y1:y1 + hight, x1:x1 + width])
            # print(str(numx))
            # print(str(numx) + '  宽:' + str(int(width)) + '    高:' + str(int(hight)))
            # cv2.imwrite('D:\\tensorflow_code\\DNA_TEST\\data\\' + str(numx) + '.jpg', original_img[y1:y1 + hight, x1:x1 + width])
        numx = numx + 1

        # cv2.rectangle(original_img, (x1, y1), (x2, y2), (55, 255, 155), 2)
    # cv2.rectangle(original_img, (x1, y1), (x2,y2), (55, 255, 155), 2)

    draw_img = cv2.drawContours(original_img.copy(), box, -1, (0, 0, 255), 4)

    return draw_img, crop_img_dist,original_img[y1:y1 + hight, x1:x1 + width]





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
    draw_img, crop_img_dist,_imgs = drawcnts_and_cut(original_img, box)

    cv2.imwrite(save_path + '\\' + img_file_name_+'.jpg', _imgs)
    fxa = 0.4
    fya = 0.4
    '''
    cv2.imshow('original_img', original_img)
    cv2.imshow('blurred', blurred)
    cv2.imshow('gradX', gradX)
    cv2.imshow('gradY', gradY)
    cv2.imshow('final', gradient)


    tempimg = cv2.resize(thresh, None, fx=fxa, fy=fya, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('thresh', tempimg)
    tempimg = cv2.resize(closed, None, fx=fxa, fy=fya, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('closed', tempimg)
    '''
    tempimg = cv2.resize(draw_img, None, fx=fxa, fy=fya, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('draw_img', tempimg)


def get_file_list_dir(img_path):
    if os.path.isfile(img_path):
        return
    fs = os.listdir(img_path)

    for f1 in fs:
        tmp_path = os.path.join(img_path,f1)
        if not os.path.isdir(tmp_path):
            if os.path.splitext(tmp_path)[1] == '.jpeg' or os.path.splitext(tmp_path)[1] == '.jpg':
                img = Image.open(tmp_path)
                if (img.size == (1600,1200) or tmp_path.find('Enhanced')>0):
                    print(f1)
                    file_analysis(img_path + "\\", f1, img_path + "\\")
        else:
            print('文件夹：',tmp_path)
            get_file_list_dir(tmp_path)
    return
    

if __name__ == '__main__':
    #file_analysis(r'F:\\PycharmProjects\\tensorflow\\TestFile\\','88142366-159b-41b7-9fc7-1326ae269714.jpeg','')
    get_file_list_dir(r'D:\tensorflow_code\DNA_TEST\s')
    cv2.waitKey(20171219)

    
