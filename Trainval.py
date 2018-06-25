#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import random
allFileNum = 0


def printPath(level, path):
     global allFileNum
     '''
     打印一个目录下的所有文件夹和文件
     '''
     # 所有文件
     fileList = []
     # 返回一个列表，其中包含在目录条目的名称(google翻译)
     files = os.listdir(path)
     # 先添加目录级别
     for f in files:
      if(os.path.isfile(path + '/' + f)):
          if (os.path.splitext(f)[1] == '.xml'):
              fileList.append(f)
              print(f)

     random.shuffle(fileList)#打乱列表
     for fl in fileList:
      # 随便计算一下有多少个文件
      allFileNum = allFileNum + 1

     #createfile(fileList,path)
     createfile_just_train(fileList,path)

#  test.txt,train.txt,val.txt,trainval.txt
def createfile(list,path):
    test = open(path + "test.txt",'w')  #测试集合 50%
    trainval = open(path + "trainval.txt",'w')  #训练和验证集合 50%
    val = open( path + "val.txt",'w')   #验证集合  25%
    train = open(path + "train.txt",'w') #训练集合 25%

    iNum = 1 #开启则写入测试集合50%
    bVorT = 0
    for fl in list:
        if (iNum == 0):
            test.writelines(os.path.splitext(fl)[0]+ '\n')
            iNum = 1
        else:
            trainval.writelines(os.path.splitext(fl)[0]+ '\n')
            iNum = 0
            if (bVorT == 0):
                val.writelines(os.path.splitext(fl)[0] + '\n')
                bVorT = 1
            else:
                train.writelines(os.path.splitext(fl)[0]+ '\n')
                bVorT = 0
    test.close();
    trainval.close();
    train.close();
    val.close();


#  train.txt,val.txt,trainval.txt
def createfile_just_train(list,path):
    trainval = open(path + "trainval.txt",'w')  #训练和验证集合 100%
    val = open( path + "val.txt",'w')   #验证集合  50%
    train = open(path + "train.txt",'w') #训练集合 50%


    bVorT = 0
    for fl in list:
        trainval.writelines(os.path.splitext(fl)[0] + '\n')
        if ((bVorT % 8)== 0 or (bVorT % 9)== 0):
            val.writelines(os.path.splitext(fl)[0] + '\n')
        else:
            train.writelines(os.path.splitext(fl)[0] + '\n')
        bVorT = bVorT+1


    trainval.close();
    train.close();
    val.close();



if __name__ == '__main__':
     printPath(1, 'D:\\tensorflow_code\\DNA_TEST\\DNA_862\\862_File\\VOC2012\\Annotations\\')
     print('总文件数 =',allFileNum )