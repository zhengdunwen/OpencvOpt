
import xml.etree.ElementTree as ET
import os
import sys

''' 
XML文件读取 
<?xml version="1.0" encoding="utf-8"?>
<catalog>
    <maxid>4</maxid>
    <login username="pytest" passwd='123456'>dasdas
        <caption>Python</caption>
        <item id="4">
            <caption>测试</caption>
        </item>
    </login>
    <item id="2">
        <caption>Zope</caption>
    </item>
</catalog>
'''


# 判断两个矩形是否相交
def mat_inter(box1, box2):
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False

# 计算两个矩形框的重合度
def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    if mat_inter(box1, box2) == True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        boxInter = [max(x01, x11)- x11,max(y01, y11)-y11,min(x02, x12)- x11,min(y02, y12)-y11]
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / area1
        return coincide,boxInter
    else:
        return 0,None


def posInRect(PosList, Rect):  # Rect [XMIN,YMIN,XMAX,YAMX]
    newList = []
    for box in PosList:
        xmin = int(box[1])
        ymin = int(box[2])
        XMAX = int(box[3])
        YMAX = int(box[4])
        rectXmin = int(Rect[0])
        rectYmin = int(Rect[1])
        rectXMAX = int(Rect[2])
        rectYMAX = int(Rect[3])

        if mat_inter((xmin,ymin,XMAX,YMAX),(rectXmin,rectYmin,rectXMAX,rectYMAX)): #相交
            coincide,boxIn = solve_coincide((xmin,ymin,XMAX,YMAX),(rectXmin,rectYmin,rectXMAX,rectYMAX))
            if coincide > 0.5: #相交的比例大于50%
                newList.append([box[0], boxIn[0],boxIn[1],boxIn[2],boxIn[3]])
        #if xmin < rectXmin or XMAX > rectXMAX or ymin < rectYmin or YMAX > rectYMAX:
        #   continue



    return newList




class XmlParse:
    def __init__(self, file_path):
        self.tree = None
        self.root = None
        self.xml_file_path = file_path

    def ReadXml(self):
        try:
            print("xmlfile:", self.xml_file_path)
            self.tree = ET.parse(self.xml_file_path)
            self.root = self.tree.getroot()
        except Exception as e:
            print("parse xml faild!")
            sys.exit()
        else:
            print("parse xml success!")
        finally:
            return self.tree

    def CreateNode(self, tag, attrib, text):
        element = ET.Element(tag, attrib)
        element.text = text
        print("tag:%s;attrib:%s;text:%s" % (tag, attrib, text))
        return element

    def AddNode(self, Parent, tag, attrib, text):
        element = self.CreateNode(tag, attrib, text)
        if Parent:
            Parent.append(element)
            el = self.root.find("lizhi")
            print(el.tag, "----", el.attrib, "----", el.text)
        else:
            print("parent is none")

    def WriteXml(self, destfile):
        dest_xml_file = os.path.abspath(destfile)
        self.tree.write(dest_xml_file, encoding="utf-8", xml_declaration=True)

    def AddBoxNode(self,boxList):
        for box in boxList:
            object = self.CreateNode('object', {}, '')

            name = self.CreateNode('name', {}, str(box[0]))
            pose = self.CreateNode('pose', {}, 'Unspecified')
            truncated = self.CreateNode('truncated', {}, '0')
            Difficult = self.CreateNode('difficult', {}, '0')
            bndbox = self.CreateNode('bndbox', {}, '')

            xmin = self.CreateNode('xmin', {}, str(box[1]))
            ymin = self.CreateNode('ymin', {}, str(box[2]))
            xmax = self.CreateNode('xmax', {}, str(box[3]))
            ymax = self.CreateNode('ymax', {}, str(box[4]))
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)

            object.append(name)
            object.append(pose)
            object.append(truncated)
            object.append(Difficult)
            object.append(bndbox)

            self.root.append(object)

    def UpdateFileInfo(self,strfilename,strWidth,strHeight):
        filename = self.CreateNode('filename', {}, strfilename)

        self.root.append(filename)
        size = self.CreateNode('size', {}, '')
        width = self.CreateNode('width', {}, strWidth)
        height = self.CreateNode('height', {},strHeight)
        depth = self.CreateNode('depth', {}, '3')
        size.append(width)
        size.append(height)
        size.append(depth)

        self.root.append(size)


    def getBoxInfo(self):
        # 修改
        objectAll = self.root.findall("object")
        listPos = []
        for object in objectAll:
            name = object.find("name").text
            bndbox = object.find("bndbox")
            xmin = bndbox.find("xmin").text
            ymin = bndbox.find("ymin").text
            xmax = bndbox.find("xmax").text
            ymax = bndbox.find("ymax").text

            listPos.append([name, xmin, ymin, xmax, ymax])
        return listPos

    def getFileInfo(self):
        filename = self.root.find("filename").text
        size = self.root.find("size")
        width = size.find("width").text
        height = size.find("height").text
        return filename,width,height


def changeBoxPos(boxList,temp,save_path,ImageFileName,XmlFileName,width,height,cutH = 640,cutW = 640,stepL = 400):
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
            imagRect = [iW,iH,cutW + iW,cutH + iH]
            newList = posInRect(boxList,imagRect)
            XMLSave = save_path + '\\segmentation_'+str(countH)+str(countW)+'_'+XmlFileName

            tmpxmliObj = XmlParse(temp)
            tmpxmliObj.ReadXml()

            ImgName = 'segmentation_'+str(countH)+str(countW)+'_'+ImageFileName
            tmpxmliObj.UpdateFileInfo(ImgName,str(cutW),str(cutH))

            tmpxmliObj.AddBoxNode(newList)

            tmpxmliObj.WriteXml(XMLSave)


def get_file_list_dir(img_path,save_path,temp):
    if os.path.isfile(img_path):
        return
    fs = os.listdir(img_path)
    bdst = 0
    for f1 in fs:
        tmp_path = os.path.join(img_path,f1)
        if not os.path.isdir(tmp_path):
            if os.path.splitext(tmp_path)[1] == '.XML' or os.path.splitext(tmp_path)[1] == '.xml':
                xmliObj = XmlParse(tmp_path)
                xmliObj.ReadXml()
                #print(xmliObj.getBoxInfo())
                ImageName, ImageW, ImageH = xmliObj.getFileInfo()

                changeBoxPos(xmliObj.getBoxInfo(),temp, save_path, ImageName,f1, int(ImageW), int(ImageH))

        else:
           # print('文件夹：',tmp_path)
            get_file_list_dir(tmp_path,save_path)
    return

if __name__ == "__main__":

    '''
    path = r"D:\DNA_BACK\100BIAOZH\0ab1a043-6307-44f8-9a66-27f11290bfd3.xml"
    temp = r"D:\DNA_BACK\100BIAOZH\temp.xml"
    tempSave = r"D:\DNA_BACK\100BIAOZH\tempSave.xml"
    xmliObj = XmlParse(path)
    xmliObj.ReadXml()
    print(xmliObj.getBoxInfo())
    ImageName,ImageW,ImageH = xmliObj.getFileInfo()

    changeBoxPos(xmliObj.getBoxInfo(),r'D:\DNA_BACK\100BIAOZH',ImageName,'0ab1a043-6307-44f8-9a66-27f11290bfd3.xml',int(ImageW),int(ImageH))
    '''
    temp = r"D:\DNA_BACK\100BIAOZH\temp.xml"
    get_file_list_dir(r'D:\DNA_BACK\100BIAOZH\xml',r'D:\DNA_BACK\100BIAOZH',temp)
