import xml.etree.ElementTree as ET
import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

if __name__ == "__main__":
    path = r"D:\DNA_BACK\100BIAOZH\segmentation_22_5a21f6c8-7cfa-4799-92e8-95cd0d8c08f3.xml"

    pathImg = r"D:\DNA_BACK\100BIAOZH\image_640\segmentation_22_5a21f6c8-7cfa-4799-92e8-95cd0d8c08f3.jpeg"

    xmliObj = XmlParse(path)
    xmliObj.ReadXml()
    listBox = xmliObj.getBoxInfo()
    print(listBox)

    img = cv2.imread(pathImg)

    for box in listBox:
        cv2.rectangle(img, (int(box[1]),int(box[2])), (int(box[3]),int(box[4])), (55, 255, 155), 2)

    cv2.imshow('brg', img)

    cv2.waitKey(20171219)

