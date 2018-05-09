import cv2
import numpy as np
import array
import matplotlib.pylab as plt
import os
import random
import pandas as pd
import Opencv.FileSearch as FS
import time


MIN_MATCH_COUNT = 4 #必须要4个点，否则无法转换映射关系




def SIFT(img_Max,img_patch,Max,Min):
    # Initiate SIFT detector

    img_patch_Y_COVER = cv2.flip(img_patch, 1)  #水平翻转
    img_1_X_COVER = cv2.cvtColor(img_patch_Y_COVER, cv2.COLOR_RGB2GRAY)
    img_1 = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)
    img_2 = cv2.cvtColor(img_Max, cv2.COLOR_RGB2GRAY)

    #create（int nfeatures = 0，int nOctaveLayers = 3，double contrastThreshold = 0.04，double edgeThreshold = 10，double sigma = 1.6）
    sift = cv2.xfeatures2d.SIFT_create()#50,3,0.04,10
    #sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1_X_COVER, des1_X_COVER = sift.detectAndCompute(img_1_X_COVER, None)
    kp1, des1 = sift.detectAndCompute(img_1,None)
    kp2, des2 = sift.detectAndCompute(img_2,None)

    #显示小图的特征点，调试开启
    #img = cv2.drawKeypoints(img_1, outImage=img_1, keypoints=kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("sift", img)

    #print("kp1_X_COVER:" + str(len(kp1_X_COVER)))
    #print("kp1:" + str(len(kp1)))
    #print("kp2:" + str(len(kp2)))

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matches_X_COVER = flann.knnMatch(des1_X_COVER, des2, k=2)
    
    '''

    # BFMatcher with default params

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    matches_X_COVER = bf.knnMatch(des1_X_COVER, des2, k=2)
    '''

    # Apply ratio test
    good_ = []
    for m,n in matches:
        if m.distance < 0.8*n.distance :
            good_.append(m)
    print("good:" +str(len(good_)))

    good_X_COVER = []
    for m,n in matches_X_COVER:
        if m.distance < 0.8*n.distance :
            good_X_COVER.append(m)
    print("good_X_COVER:" +str(len(good_X_COVER)))


    #比较good的点的比例，高的则适配更好
    if len(matches_X_COVER) > 0:
        _X_Cover_per = float( len(good_X_COVER)/len(matches_X_COVER))
    else:
        _X_Cover_per = 0

    if len(matches) >0:
        _per = float(len(good_) / len(matches))
    else:
        _per = 0


    if _X_Cover_per > _per:
        good = good_X_COVER.copy()
        src_pts = np.float32([ kp1_X_COVER[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        print("_X_Cover_per:" + str(_X_Cover_per))
    else:
        good = good_.copy()
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        print("_per:" + str(_per))



    dst_pts,src_pts = MeanCheck(dst_pts,src_pts)
    #print(dst_pts)
    print("good check:" + str(len(dst_pts)))





    if len(dst_pts)>=MIN_MATCH_COUNT:
        # 获取关键点的坐标
        #src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # 第三个参数 Method used to computed a homography matrix. The following methods are possible:
        #0 - a regular method using all the points
        #CV_RANSAC - RANSAC-based robust method
        #CV_LMEDS - Least-Median robust method
        # 第四个参数取值范围在 1 到 10 , 绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
        # 超过误差就认为是 outlier
        # 返回值中 H         为变换矩阵。
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        #wrap = cv2.warpPerspective(img2, H, (img2.shape[1] + img2.shape[1], img2.shape[0] + img2.shape[0]))
        #wrap[0:img2.shape[0], 0:img2.shape[1]] = img_patch


        #matchesMask = mask.ravel().tolist()
#        # 获得原图像的高和宽
        h,w,_ = img_patch.copy().shape
        print("h:" +str(h)  + "  w:"+str(w) )
#        # 使用得到的变换矩阵对原图像的四个角进行变换,获得在目标图像上对应的坐标。
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        if H is not None:
          dst = cv2.perspectiveTransform(pts,H)
          dst, newH_W = boxCheck(dst)
          if newH_W[0] * newH_W[1] > h * w * Max :
                dst = []
                print("面积大于阈值匹配失败",newH_W[0] * newH_W[1],h * w*Max)
          if newH_W[0] * newH_W[1] < h * w * Min:
                dst = []
                print("面积小于阈值匹配失败",newH_W[0] * newH_W[1],h * w*Min)
          # 在原图中画出目标所在位置框, cv2.LINE_AA表示闭合框
          else:
            c = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
            cv2.polylines(img_Max,[np.int32(dst)],True,c,2, cv2.LINE_4)
        else:
            dst = []



    elif len(dst_pts)>=1:
        print("小特征点匹配")
        # 获取关键点的坐标
        #src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        if len(dst_pts) < 3:
            x_mid = 0
            y_mid = 0
            for dst_m in dst_pts:
                x_mid = x_mid + dst_m[0][0]
                y_mid = y_mid + dst_m[0][1]

            y_mid = y_mid / len(dst_pts)
            x_mid = x_mid / len(dst_pts)

        elif len(dst_pts) >= 3:
            m = cv2.moments(dst_pts)
            if m["m00"] != 0.0:
                 x_mid= int(m["m10"]/m["m00"])
                 y_mid= int(m["m01"]/m["m00"])
            else:
                x_mid = 0
                y_mid = 0
                for dst_m in dst_pts:
                    x_mid = x_mid + dst_m[0][0]
                    y_mid = y_mid + dst_m[0][1]

                y_mid = y_mid / len(dst_pts)
                x_mid = x_mid / len(dst_pts)

        # 获得原图像的高和宽
        h, w, _ = img_patch.shape
        WHmax = (h + w) / 2
        a1 = [[[x_mid + WHmax, y_mid - WHmax]], [[x_mid - WHmax, y_mid - WHmax]], [[x_mid - WHmax, y_mid + WHmax]],
              [[x_mid + WHmax, y_mid + WHmax]]]
        #        # 使用得到的变换矩阵对原图像的四个角进行变换,获得在目标图像上对应的坐标。
        #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        #dst = cv2.perspectiveTransform(pts, a1)

        #print(dst.shape)
        dst,newH_W = boxCheck(a1)
        '''
        if newH_W[0] * newH_W[1] > h * w * Max:
            dst = []
            print("面积大于阈值匹配失败", newH_W[0] * newH_W[1], h * w * Max)
        '''
        if newH_W[0] * newH_W[1] < h * w * Min:
            dst = []
            print("面积小于阈值匹配失败", newH_W[0] * newH_W[1], h * w * Min)
        # 在原图中画出目标所在位置框, cv2.LINE_AA表示闭合框
        else:
            c = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
            cv2.polylines(img_Max, [np.int32(dst)], True, c, 2, cv2.LINE_4)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        dst= []

#    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#    singlePointColor = None,
#    matchesMask = matchesMask, # draw only inliers
#    flags = 2)
#    img3 = cv2.drawMatches(img_patch,kp1,img_Max,kp2,good,None,**draw_params)
#    img3 = cv2.drawMatches(img_patch,kp1,img_Max,kp2,good,None,matchesMask = matchesMask,flags=2)

    good = np.expand_dims(good,1)


    img3 = cv2.drawMatchesKnn(img_patch,kp1,img_Max,kp2,good[:20],None, flags=2)


    fx = 0.4
    fy = 0.4
    img3 = cv2.resize(img3, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    return img3,dst


def boxCheck(box):

    Xs = []
    Xs.append(box[0][0][0])
    Xs.append(box[1][0][0])
    Xs.append(box[2][0][0])
    Xs.append(box[3][0][0])

    Ys =[]
    Ys.append(box[0][0][1])
    Ys.append(box[1][0][1])
    Ys.append(box[2][0][1])
    Ys.append(box[3][0][1])

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
    print("new h:" + str(h) + "  w:" + str(w))
    xBox = np.ndarray(shape=(4,1, 2), dtype=float, buffer=np.array([x2, y1,x1, y1,x1, y2,x2, y2,0]), offset=0, order="C")
    HandW = [h,w]
    return xBox,HandW


def MeanCheck(des,src):
    Xs = []
    Ys = []
    for i in range(0,len(des)):
        Xs.append(des[i][0][0])
        Ys.append(des[i][0][1])

    dataX= pd.DataFrame(Xs)
    dataY = pd.DataFrame(Ys)

    sigma_X = dataX.std()
    mean_x = dataX.mean()


    delV = []
    for x in range(0,len(Xs)):
        if abs(Xs[x] - float(mean_x)) > float(sigma_X) * 1.4:
            delV.append(x)


    sigma_Y = dataY.std()
    mean_Y = dataY.mean()


    for y in range(0,len(Ys)):
        if abs(Ys[y] - float(mean_Y)) > float(sigma_Y) * 1.4:
            delV.append(y)

    delV2 = list(set(delV))


    des = np.delete(des,delV2,0)
    src = np.delete(src,delV2,0)


    return des,src



def get_file_list(img_path,imgSrc):
    if os.path.isfile(img_path):
        return [],[]

    group_info = os.walk(img_path)
    posall = []
    posFileNum = []

    for _,_,files in group_info:
        for filename in files:
            fn = os.path.splitext(filename)
            if fn[1] == '.jpg':
                img1 = cv2.imread(img_path + filename)
                print(filename)
                _,pos = SIFT(imgSrc.copy(),img1,4,0.6)
                if len(pos) > 0:
                    posall.append(pos)
                    posFileNum.append(filename)

        return posall,posFileNum
    return posall,posFileNum



#文件和文件的比较逻辑
def cmp_file_and_file(_src,_dst,_save=''):
    #img_dst= cv2.imread(r'D:\tensorflow_code\DNA_TEST\862_File_cut\b291a960-a6e9-490e-a70f-3dc613c38564\b291a960-a6e9-490e-a70f-3dc613c38564+20+0+0.jpg')  # queryImage
    #img_src = cv2.imread( r'D:\tensorflow_code\DNA_TEST\DNA_862\862_File\KSDAPB180329-132\3a7fc5f7-e011-49df-a1fe-13991c500fdc.jpeg')  # trainImage

    img_src = cv2.imread(_src)
    img_dst= cv2.imread(_dst)

    #ret, img_src = cv2.threshold(img_src, 80, 255, cv2.THRESH_TOZERO_INV)
    #ret, img_dst = cv2.threshold(img_dst, 80, 255, cv2.THRESH_TOZERO_INV)

    result, boxs = SIFT(img_src, img_dst, 4, 0.6)
    cv2.imshow("re", result)
    if _save != '':
        cv2.imwrite(_save, result)


#文件夹的比较逻辑
def cmp_path():
    imgSrc = cv2.imread(r'D:\tensorflow_code\DNA_TEST\DNA_862\862_File\KSDAPB180329-132\3a7fc5f7-e011-49df-a1fe-13991c500fdc.jpeg')
    posall,posFileNum = get_file_list('D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\b291a960-a6e9-490e-a70f-3dc613c38564\\',imgSrc)

    width = 3000
    height = 1500

    resultImg = np.zeros((height, width, 3), dtype=np.uint8)
    resultImg = 255 - resultImg

    ims = imgSrc.copy()
    for pos_ in range(0, len(posall)):
        c = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))

        cv2.rectangle(ims,
                      (int(posall[pos_][1][0][0]), int(posall[pos_][1][0][1])),
                      (int(posall[pos_][3][0][0]), int(posall[pos_][3][0][1])),
                      c,
                      2)
        #+16+1+0.jpg
        strFilename = str(posFileNum[pos_])
        code1 = strFilename.find('+')
        code2 = strFilename.find('+',code1+1)
        code3 = strFilename.find('+', code2+1)
        strId = strFilename[code1+1:code2]
        strLR = strFilename[code2+1:code3]

        imgDst = imgSrc[int(posall[pos_][1][0][1]): int(posall[pos_][3][0][1]),int(posall[pos_][1][0][0]):int(posall[pos_][3][0][0])]  # 截取第5行到89行的第500列到630列的区域
        PingJieTuPian(resultImg,imgDst,int(strId),int(strLR))





    fx = 1
    fy = 1
    img3 = cv2.resize(ims, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\Xcmp.jpg', img3)
    cv2.imwrite('D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\Xcmp_t.jpg', resultImg)
    cv2.imshow('result.py', img3)
    cv2.imshow("a", resultImg)


def PingJieTuPian(imgSrc,imgDst,id,idLR):
    id =(id*2)-idLR-1
    idRow = id %10
    idCow= id //10
    h, w, _ = imgDst.shape

    y1 = idCow*300
    y2 = idCow*300 + h

    x1 = idRow*300
    x2 = idRow*300+w

    imgSrc[y1:y2,x1:x2] = imgDst  # 指定位置填充，大小要一样才能填充




if __name__ == '__main__':
    #cmp_path()

    img_src = 'D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\gama_img0.jpg'
    img_dst= 'D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\gama.jpg'
    img_save= 'D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\xm.jpg'
    cmp_file_and_file(img_src,img_dst,img_save)


    '''
    FileList = FS.getFileAndSourc(r'D:\tensorflow_code\DNA_TEST\DNA_862\862_File\test.txt')
    pathCut = "D:\\tensorflow_code\\DNA_TEST\\862_File_cut\\H"

    fo = open("D:\\Test\\test.txt", "w")

    for dist_ in FileList:
        path = dist_['path']
        src = dist_['src']
        dst = dist_['dst']

        for num in range(0,2):

            strDst = pathCut + "\\" + dst[0:-5]+'+2+'+str(num)+'+0.jpg'
            print(strDst)
            strSrc = path +"\\"+ src
            print(strSrc)

            if os.path.exists(strDst) and os.path.exists(strSrc):
                img_src = cv2.imread(strSrc)
                img_dst = cv2.imread(strDst)

                result,boxs = SIFT(img_src,img_dst,4,0.6)
                fx = 0.8
                fy = 0.8
                result = cv2.resize(result, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                #cv2.imwrite("D:\\Test\\" + dst[0:-5]+'+2+'+str(num)+'+0.jpg',result)
                fo.writelines( strSrc +"\n" +str(num) + "\n"+ str(boxs)+"\n")


    '''


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
