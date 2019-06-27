#Autor Satoshi Ishibushi
# -*- coding: utf-8 -*-
import sys,os
from PyQt4 import *
from PyQt4 import QtGui
from PyQt4 import QtCore 
from PyQt4.QtGui import *
import datetime
from PyQt4.QtCore import *
import codecs
import numpy as np
import math
import yaml
import glob
import cv2
import os
dataset=sys.argv[1]
clip_num=sys.argv[2]
env_para=np.genfromtxt(dataset+"/Environment_parameter.txt",dtype= None,delimiter =" ")
DATA_initial_index= int(env_para[5][1]) #Initial data num
DATA_last_index= int(env_para[6][1]) #Last data num
data_num =DATA_last_index - DATA_initial_index +1


Env="em"
init_num=DATA_initial_index

click =0

data_cut=40
name_list="place_name.txt"
func_list="description_function.txt"
if clip_num =="tsubo":
    map_y0=300
    map_y1=800
    map_x0=0
    map_x1=500
    clip=1
elif clip_num=="wada":
    map_y0=100
    map_y1=650
    map_x0=100
    map_x1=570
    clip=1
elif clip_num=="kitano":
    map_y0=200
    map_y1=500
    map_x0=100
    map_x1=570
    clip=1
else:
    clip=0
check_list_name=[]
check_list_func=[]

#QtCore.QTextCodec.setCodecForCStrings(QtCore.QTextCodec.codecForName('unicode'))
def map_point(num):
    yam=glob.glob(dataset+"/map/*.yaml")
    f = open(yam[0], 'r')
    ya=yaml.load(f)
    f.close()

    img=cv2.imread(dataset+"/map/"+ya["image"])

    pose=np.loadtxt(dataset+"/position_data/"+repr(num)+".txt")
    pose=pose[0]
    pose[0]=round(pose[0],1)
    pose[1]=round(pose[1],1)

    x_d=math.fabs((ya["origin"][0]-pose[0])/0.05)
    y_d=math.fabs((ya["origin"][1]-pose[1])/0.05)
    #print x_d
    #print y_d

    x=int(x_d)
    y=int(img.shape[0]-y_d)
    #print len(img[0])
    for i in range(5):
        img[y+i][x-2]=[0,0,255]
        img[y+i][x-1]=[0,0,255]
        img[y+i][x]=[0,0,255]
        img[y+i][x+1]=[0,0,255]
        img[y+i][x+2]=[0,0,255]
    if clip==1:    
        img=img[map_y0:map_y1, map_x0:map_x1]  
    return img

class MainMenu(QtGui.QWidget):
    def __init__(self,parent=None):
        self.num=int(np.random.uniform(init_num,init_num+data_cut))
        QtGui.QWidget.__init__(self, parent=parent)

        name_x=50
        text_y=650
        list_y=550
        self.Label_name = QtGui.QLabel(u'ここの場所の名前を教えてください?\n選択または記入してください．',self)
        self.Label_name.setGeometry(name_x,500,400,40)

        func_x=600
        self.Label_func = QtGui.QLabel(u'何をする場所ですか?(動作,作業,目的)\n選択または記入してください．',self)
        self.Label_func.setGeometry(func_x,500,400,40)

        save_button_name=QtGui.QPushButton(u'答える',self)
        save_button_name.setGeometry(150,580,100,80)
        self.connect(save_button_name,QtCore.SIGNAL('clicked()'),self.save_name)

        save_button_func=QtGui.QPushButton(u'答える',self)
        save_button_func.setGeometry(800,580,100,80)
        self.connect(save_button_func,QtCore.SIGNAL('clicked()'),self.save_function)

        button = QtGui.QPushButton('Next',self)
        button.setGeometry(1100,650,100,80)
        self.connect(button,QtCore.SIGNAL('clicked()'),self.changeImage)

        self.Label3 = QtGui.QLabel(u'赤い点が今のあなたの位置です．',self)
        self.Label3.setGeometry(30,20,350,40)

        self.pic=QtGui.QLabel(window)
        self.pic.setGeometry(580, 0, 650, 500)
        img=dataset+"/image/"+repr(self.num)+".jpg"
        self.pic.setPixmap(QtGui.QPixmap(img))
        
        self.map=QtGui.QLabel(window)
        self.map.setGeometry(30,50, 500, 400)
        #img=dataset+"/image/"+repr(num)+".jpg"
        map=map_point(self.num)
        cv2.imwrite("map.jpg",map)
        #print img
        self.map.setPixmap(QtGui.QPixmap("map.jpg"))


        #self.w.show()

        self.show()



    def changeImage(self):
        global click
        click +=1
        self.num=int(np.random.uniform(init_num+(data_cut*click),init_num+(data_cut*(click+1))))
        if self.num>data_num:
            os.remove("map.jpg")
            sys.exit()        

        img=dataset+"/image/"+repr(self.num)+".jpg"
        self.pic.setPixmap(QtGui.QPixmap(img))
        print self.num,".jpg"
        #ind=self.combo.currentIndex()
        #print self.lines[ind]

        map=map_point(self.num)
        cv2.imwrite("map.jpg",map)
        #print img
        self.map.setPixmap(QtGui.QPixmap("map.jpg"))


    def save_name(self):

        self.w = QtGui.QWidget()
        
        self.sub = SubWindow_name(self.w,self.num)
        self.sub.show()
        QMessageBox.about(self.w, "Message", u"保存されました．")

    def save_function(self):
        self.w = QtGui.QWidget()
        self.sub = SubWindow_func(self.w,self.num)
        self.sub.show()
        QMessageBox.about(self.w, "Message", u"保存されました．")        


class SubWindow_name():

    def __init__(self,parent,data_num):
        #print data_num
        self.num=data_num
        self.w = QtGui.QDialog(parent)

        label = QtGui.QLabel(self.w)
        label.setText(u'ここの場所の名前を教えてください? \n当てはまるものにすべてチェックしてください．')
        label.setGeometry(10, 10, 300, 80)

        label_add = QtGui.QLabel(self.w)
        label_add.setText(u'追加したい場合は入力して，ボタンをおしてください．') 
        
        add_button=QtGui.QPushButton(u'+ 追加',self.w) 
        save_button=QtGui.QPushButton(u'保存',self.w)    
        #self.check=QCheckBox(self.w)
        layout = QVBoxLayout(self.w)
        
        f=codecs.open(name_list,"r",'utf-8')
        lines = f.readlines()
        f.close()


        self.checks = []
        self.name =[]
        n=-1
        i=0
        size_x=200
        for l in lines:
            if (i%10)==0:
                j=0
                n+=1
            x=30+(size_x*n)
            y=80+(j*50)
            c = QCheckBox(u"{0}".format(l),self.w)

            c.setGeometry(x, y, size_x, 30)
            global check_list_name
            if i in check_list_name:
                c.setChecked(True)
            j+=1
            #layout.addWidget(c)
            self.checks.append(c)
            self.name.append(l)
            i +=1
        y=80+(10*50)
        label_add.setGeometry(50, y+60, 350, 50)
        
        self.text_name=QLineEdit(self.w)
        self.text_name.setGeometry(50, y+100, 300, 50)
        
        add_button.setGeometry(360,y+100,50,50)
        save_button.setGeometry(430,y+100,80,80)
        self.w.connect(add_button,QtCore.SIGNAL('clicked()'),self.add)
        self.w.connect(save_button,QtCore.SIGNAL('clicked()'),self.save)
        wind=x+size_x
        if (wind)<500:
            wind =550
        self.w.setGeometry(200, 0,wind , y+200)
    def add(self):
        text_ans = self.text_name.text()
        if text_ans=="" or text_ans==" " or text_ans=="　":
            QMessageBox.about(self.w, "Message", u"入力してください．")
        else:
            name_file= codecs.open(name_list,"a",'utf-8')
            name_file.write(u"{0}\n".format(text_ans))
        global check_list_name
        check_list_name=[]
        for c in range(len(self.checks)):
            if self.checks[c].isChecked() == True:
                print c
                check_list_name.append(c)
        self.w.close()
        self.sub = SubWindow_name(self.w,self.num)
        self.sub.show()
    def save(self):
        file_name = "data/"+Env+"/name/"+repr(self.num)+".txt"
        f = codecs.open(file_name,"a",'utf-8')
        n=0
        global check_list_name
        check_list_name=[]
        for c in range(len(self.checks)):
            check=self.checks[c]
            if check.isChecked() == True:
                n+=1
                print(u"{0}".format(self.name[c]))
                f.write(u"{0}\n".format(self.name[c]))
                check_list_name.append(c)
        f.close()
        if n==0:
            QMessageBox.about(self.w, "Message", u"最低一つチェックしてください．\nなければ追加してください．")
        else:
            self.w.close()
    def show(self):
        self.w.exec_()


class SubWindow_func():

    def __init__(self,parent,data_num):
        #print data_num
        self.num=data_num
        self.w = QtGui.QDialog(parent)

        label = QtGui.QLabel(self.w)
        label.setText(u'何をする場所ですか?(動作,作業,目的)\n選択または記入してください．')
        label.setGeometry(10, 10, 300, 80)

        label_add = QtGui.QLabel(self.w)
        label_add.setText(u'追加したい場合は入力して，ボタンをおしてください．') 
        
        add_button=QtGui.QPushButton(u'+ 追加',self.w) 
        save_button=QtGui.QPushButton(u'保存',self.w)    
        #self.check=QCheckBox(self.w)
        layout = QVBoxLayout(self.w)
        
        f=codecs.open(func_list,"r",'utf-8')
        lines = f.readlines()
        f.close()


        self.checks = []
        self.name =[]
        n=-1
        i=0
        size_x=200
        for i,l in enumerate(lines):
            if (i%10)==0:
                j=0
                n+=1
            x=30+(size_x*n)
            y=80+(j*50)
            c = QCheckBox(u"{0}".format(l),self.w)
            c.setGeometry(x, y, size_x, 30)
            global check_list_func
            if i in check_list_func:
                c.setChecked(True)
            j+=1
            #layout.addWidget(c)
            self.checks.append(c)
            self.name.append(l)
            i +=1
        y=80+(10*50)
        label_add.setGeometry(50, y+60, 350, 50)
        
        self.text_name=QLineEdit(self.w)
        self.text_name.setGeometry(50, y+100, 300, 50)
        
        add_button.setGeometry(360,y+100,50,50)
        save_button.setGeometry(430,y+100,80,80)
        self.w.connect(add_button,QtCore.SIGNAL('clicked()'),self.add)
        self.w.connect(save_button,QtCore.SIGNAL('clicked()'),self.save)
        wind=x+size_x
        if (wind)<500:
            wind =550
        self.w.setGeometry(200, 0,wind , y+200)
    def add(self):
        text_ans = self.text_name.text()
        if text_ans=="" or text_ans==" " or text_ans=="　":
            QMessageBox.about(self.w, "Message", u"入力してください．")
        else:
            func_file= codecs.open(func_list,"a",'utf-8')
            func_file.write(u"{0}\n".format(text_ans))
        global check_list_func
        check_list_func=[]
        for c in range(len(self.checks)):
            if self.checks[c].isChecked() == True:
                print c
                check_list_func.append(c)

        self.w.close()
        self.sub = SubWindow_func(self.w,self.num)
        self.sub.show()
    def save(self):
        file_name = "data/"+Env+"/function/"+repr(self.num)+".txt"
        f = codecs.open(file_name,"a",'utf-8')
        n=0
        global check_list_func
        check_list_func=[]
        for c in range(len(self.checks)):
            check=self.checks[c]
            if check.isChecked() == True:
                n+=1
                print(u"{0}".format(self.name[c]))
                f.write(u"{0}\n".format(self.name[c]))
                check_list_func.append(c)
        f.close()
        if n==0:
            QMessageBox.about(self.w, "Message", u"最低一つチェックしてください．\nなければ追加してください．")
        else:
            self.w.close()
    def show(self):
        self.w.exec_()







app = QtGui.QApplication(sys.argv)
window = QtGui.QMainWindow()
window.setGeometry(0, 0, 1300, 750)
window.setWindowTitle('Plac_Concept Data Inputer')    

main = MainMenu()
window.setCentralWidget(main)
window.show()
#window.w=QtGui.QWidget()
#text=window.set
"""
combo=QComboBox(window)
combo.setGeometry(700, 300, 300, 50)
#t="You are win!"
#combo.addItem(t)
word_file="dictionary.txt"
#word_file="a.txt"
f=codecs.open(word_file,"r",'utf-8')
lines = f.readlines()
f.close()
for l in lines:
    #l.encode('utf-8')
    #l = l[:-1].split('\n')

    print l
    #print u"{0}".format(l)
    combo.addItem(u"{0}".format(l))
"""
#pic = QtGui.QLabel(window)
#pic.setGeometry(10, 0, 650, 500)
#use full ABSOLUTE path to the image, not relative
#pic.setPixmap(QtGui.QPixmap(os.getcwd() + "/1.jpg"))



sys.exit(app.exec_())
