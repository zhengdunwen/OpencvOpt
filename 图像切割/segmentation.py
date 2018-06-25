import cv2
from math import *
import numpy as np
import os
import time
from PIL import Image


def cut_image(img_path, img_file_name_, save_path,cutW,cutH,stepL):
    image = cv2.imread(img_path + img_file_name_)
    height ,width=image.shape[:2]

    #小图扩充成标准图
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1)
    if width < cutW and height < cutH:
        image = cv2.warpAffine(image, matRotation, (cutW, cutH), borderValue=(255, 255, 255))
        width = cutW
        height = cutH
    elif width < cutW:
        if height > cutH:
            image = cv2.warpAffine(image, matRotation, (cutW, height), borderValue=(255, 255, 255))
            width = cutW
        else:
            image = cv2.warpAffine(image, matRotation, (cutW, cutH), borderValue=(255, 255, 255))
            width = cutW
            height = cutH
    elif height < cutH:
        if width > cutW:
            image = cv2.warpAffine(image, matRotation, (width, cutH), borderValue=(255, 255, 255))
            height = cutH
        else:
            image = cv2.warpAffine(image, matRotation, (cutW, cutH), borderValue=(255, 255, 255))
            width = cutW
            height = cutH

    print(str(height) +' w'+str(width))
    wSize = 0
    hSize = 0

    wList = []
    while wSize <= width:
        wList.append(wSize)
        wSize = wSize + stepL
        if (wSize + cutW) > width:
            wSize = width -cutW
            wList.append(wSize)
            break

    hList = []
    while hSize <= height:
        hList.append(hSize)
        hSize = hSize + stepL
        if (hSize + cutH) > height:
            hSize = height - cutH
            hList.append(hSize)
            break


    countH = 0
    for iH in hList:
        countH += 1
        countW = 0
        for iW in wList:
            countW+=1
            imagePis = image[iH:cutH+iH,iW:cutW + iW]
            name = save_path + '\\segmentation_'+str(countH)+str(countW)+'_'+img_file_name_
            cv2.imwrite(name,imagePis)



def get_file_list_dir(img_path):
    if os.path.isfile(img_path):
        return
    fs = os.listdir(img_path)
    num = 0
    for f1 in fs:
        tmp_path = os.path.join(img_path,f1)
        if not os.path.isdir(tmp_path):
            if os.path.splitext(tmp_path)[1] == '.jpeg' or os.path.splitext(tmp_path)[1] == '.jpg':
                if (tmp_path.find('reduce_') > 0):
                    #print(f1)
                    cut_image(img_path + "\\", f1, img_path + "\\",640,640,400)
                    num = num +1
                    print(str(num))
        else:
            #print('文件夹：',tmp_path)
            get_file_list_dir(tmp_path)
    return

if __name__ == '__main__':
    #单个文件验证代码
    get_file_list_dir(r'D:\DNA_BACK\3000DNA\\')

