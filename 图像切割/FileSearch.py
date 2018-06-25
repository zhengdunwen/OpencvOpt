import os
import sys
from PIL import Image
import shutil
import glob

rootdir = r'D:\DNA_BACK\3000DNA'

def files(curr_dir = '.', ext = '*.exe'):
    """当前目录下的文件"""
    for i in glob.glob(os.path.join(curr_dir, ext)):
        yield i

src_map ={}
# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        print(child)

def file_extension(path):
  return os.path.splitext(path)[1]

def file_getFileNmae(path):
    return os.path.basename(path)


def traverse(f,fo):
    fs = os.listdir(f)
    toWrit = 'path'+'\n'
    toWrit = toWrit + f + '\n'
    bsrc = 0
    bdst = 0
    for f1 in fs:
        tmp_path = os.path.join(f,f1)

        if not os.path.isdir(tmp_path):
            if file_extension(tmp_path) == '.jpeg' or file_extension(tmp_path) == '.jpg':
                img = Image.open(tmp_path)
                if (img.size == (1600,1200) or tmp_path.find('Enhanced')>0) and bsrc ==0:
                    toWrit = toWrit +'src'+'\n'
                    toWrit = toWrit +file_getFileNmae(tmp_path)+'\n'
                    bsrc = 1
                elif (img.size == (1232,912) or tmp_path.find('Karyotype')>0) and bdst == 0:
                    toWrit = toWrit + 'dst'+'\n'
                    toWrit = toWrit + file_getFileNmae(tmp_path)+'\n'
                    bdst = 1

        else:
            print('文件夹：',tmp_path)
            traverse(tmp_path,fo)
    fo.writelines(toWrit)


def getFileAndSourc(path):
    list = []
    fread = open(path, "r")

    strinfo = fread.readline()
    info = {}
    while (len(strinfo) > 0):
        if "path" in strinfo:
            path = fread.readline()
            info['path'] = path[0:-1]
        elif "src" in strinfo:
            src = fread.readline()
            info['src'] = src[0:-1]
        elif "dst" in strinfo:
            dst = fread.readline()
            info['dst'] = dst[0:-1]
        strinfo = fread.readline()

        if len(info) ==3:
            list.append(info.copy())
            info.clear()


    return  list



def traverse_del(f):
    fs = os.listdir(f)
    filenum = 0
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            if file_extension(tmp_path) == '.jpg':
                filenum = 3
        else:
            traverse_del(tmp_path)

    if filenum > 2:
        shutil.rmtree(f)
        print("delete:",f)



#删除多余的文件
def remove_most_files(img_path):
    if os.path.isfile(img_path):
        return
    fs = os.listdir(img_path)
    bdst = 0
    bsrc = 0
    for f1 in fs:
        tmp_path = os.path.join(img_path,f1)
        if not os.path.isdir(tmp_path):
            if os.path.splitext(tmp_path)[1] == '.jpeg' or os.path.splitext(tmp_path)[1] == '.jpg':
                fp = open(tmp_path, 'rb')
                img = Image.open(fp)
                if (img.size == (1232,912)):
                    bdst = bdst +1
                    if bdst > 1:
                        fp.close()
                        os.remove(tmp_path)
                elif (img.size == (1600,1200)):
                    bsrc = bsrc +1
                    if bsrc > 1:
                        fp.close()
                        os.remove(tmp_path)
        else:
           # print('文件夹：',tmp_path)
           remove_most_files(tmp_path)
    return


def deletefile(path):
    #文件导入文件验证
    fo = open("C:\\Users\\zdw\\Desktop\\new2.txt", "r")
    line = fo.readline()
    filelist = []
    while len(line) > 0:
        linepos = line.find(": ")
        filename = line[linepos+2:-6]
        filelist.append(filename)
        line = fo.readline()
    fs = os.listdir(path)
    filenum = 0
    for f1 in fs:
        tmp_path = os.path.join(path,f1)
        for filename in filelist:
            pos = f1.find(filename)
            if  pos > -1 and os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(tmp_path)




if __name__ == '__main__':

    #fo = open(rootdir + "\\fileSrcAndDst.txt", "w")
    #filePathC = rootdir
    # traverse(filePathC,fo)


    #getFileAndSourc(rootdir + "\\test.txt")


    #traverse_del(rootdir)
    deletefile(r"D:\DNA_BACK\3000_cut5")
    #remove_most_files(r'D:\DNA_BACK\3000DNA')
