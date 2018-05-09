import  os
from PIL import Image
import shutil
rootdir = r'D:\tensorflow_code\DNA_TEST\DNA_862\862_File'


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

    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            if file_extension(tmp_path) == '.jpeg':
                img = Image.open(tmp_path)
                if img.size == (1600,1200):
                    toWrit = toWrit +'src'+'\n'
                    toWrit = toWrit +file_getFileNmae(tmp_path)+'\n'
                elif img.size == (1232,912):
                    toWrit = toWrit + 'dst'+'\n'
                    toWrit = toWrit + file_getFileNmae(tmp_path)+'\n'

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


if __name__ == '__main__':

    fo = open(rootdir + "\\test.txt", "w")
    filePathC = rootdir
    traverse(filePathC,fo)
    getFileAndSourc(rootdir + "\\test.txt")
    '''
    traverse_del(rootdir)
    '''
