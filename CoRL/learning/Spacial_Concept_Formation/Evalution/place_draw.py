#!/usr/bin/env python
# -*- coding:utf-8 -*-
#学習した場所領域のサンプルをrviz上に可視化するプログラム
#作成者　石伏智
#作成日 2015年12月

import glob
import re
import os
import rospy
import math
import sys
import geometry_msgs.msg as gm
from geometry_msgs.msg import Point
import sensor_msgs.msg as sm
from  visualization_msgs.msg import Marker
from  visualization_msgs.msg import MarkerArray
import numpy as np
import struct           
sys.path.append("../lib/")
import file_read as f_r

RAD_90=math.radians(90)
color_all=1 #1 or 0 、0ならばすべて赤
mu_draw =1#1 or 0 、0ならば中心値を表示しない
sample_draw=1 #1 or 0,0ならばサンプルを表示しない
arrow=0
COLOR=[
[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0.5,0,0.5], #4
[0,0.5,0.5],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.6,0.2,0.2],#9
[0.2,0.6,0.2],[0.2,0.2,0.6],[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4], #14
[0.7,0.2,0.1],[0.7,0.1,0.2],[0.2,0.7,0.1],[0.1,0.7,0.2],[0.2,0.1,0.7],#19
[0.1,0.2,0.7],[0.5,0.2,0.3],[0.5,0.3,0.2],[0.3,0.5,0.2],[0.2,0.5,0.3],#24
[0.3,0.2,0.5],[0.2,0.3,0.5],[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7],#29
[0.6,0.3,0.1],[0.6,0.1,0.3],[0.1,0.6,0.3],[0.3,0.6,0.1],[0.3,0.1,0.6],#34
[0.1,0.3,0.6],[0.8,0.2,0],[0.8,0,0.2],[0.2,0.8,0],[0,0.8,0.2],#39
[0.2,0,0.8],[0,0.2,0.8],[0.7,0.3,0],[0.7,0,0.3],[0.3,0.7,0.0],#44
[0.3,0,0.7],[0,0.7,0.3],[0,0.3,0.7],[0.25,0.25,0.5],[0.25,0.5,0.25], #49
[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0.5,0,0.5], #54
[0,0.5,0.5],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.6,0.2,0.2],#59
[0.2,0.6,0.2],[0.2,0.2,0.6],[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4], #64
[0,7,0.2,0.1],[0.7,0.1,0.2],[0.2,0.7,0.1],[0.1,0.7,0.2],[0.2,0.1,0.7],#69
[0.1,0.2,0.7],[0.5,0.2,0.3],[0.5,0.3,0.2],[0.3,0.5,0.2],[0.2,0.5,0.3],#74
[0.3,0.2,0.5],[0.2,0.3,0.5],[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7],#79
[0.6,0.3,0.1],[0.6,0.1,0.3],[0.1,0.6,0.3],[0.3,0.6,0.1],[0.3,0.1,0.6],#84
[0.1,0.3,0.6],[0.8,0.2,0],[0.8,0,0.2],[0.2,0.8,0],[0,0.8,0.2],#89
[0.2,0,0.8],[0,0.2,0.8],[0.7,0.3,0],[0.7,0,0.3],[0.3,0.7,0.0],#94
[0.3,0,0.7],[0,0.7,0.3],[0,0.3,0.7],[0.25,0.25,0.5],[0.25,0.5,0.25] #99
]
Parameter_diric=sys.argv[1]
try: 
    Number=int(sys.argv[2])
except IndexError:
    Number=None
env_para=np.genfromtxt(Parameter_diric+"/Parameter.txt",dtype= None,delimiter =" ")
Class_NUM=80#nt(env_para[4][1])
#=============書く場所領域に割り当てられているデータの読みこみ===================
def class_check():
    class_count=[0.0 for i in range(Class_NUM)]
    class_list=np.loadtxt(Parameter_diric+"class.txt")
    not_class=[]
    print class_list
    for d in class_list:
        class_list[d] +=1.0
    for c in range(Class_NUM):
        if class_count[c]==1.0:
            not_class.append(c)


    return not_class

def place_draw():

    no_data_class=class_check()



    pub = rospy.Publisher('draw_space',MarkerArray)
    rospy.init_node('draw_spatial_concepts', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    sample=f_r.sampling_read(Parameter_diric)
    mu_all=f_r.mu_read(Parameter_diric)
    #sample=f_r.sampling_read(Parameter_diric)#position_read()
    marker_array=MarkerArray()
    id=0
    for c in range(Class_NUM):
        #場所領域の中心値を示す場合
 

        #===場所領域の範囲の可視化====================
        if sample_draw==1:
            for i in range(len(sample[0])):
                marker =Marker()
                if arrow==1: #矢印を可視化する場合
                    marker.type=Marker.ARROW

                    orient_sin=sample[c][i][2]
                    orient_cos=sample[c][i][3]
                    #sinを-1<=x<=1に変換
                    if orient_sin>1.0:
                        orient_sin=1.0
                    elif orient_sin<-1.0:
                        orient_sin=-1.0

                    #radian xを導出
                    radian=math.asin(orient_sin)
                    if orient_sin>0 and orient_cos<0:
                        radian=radian+RAD_90
                    elif orient_sin<0 and orient_cos<0:
                        radian=radian-RAD_90
                
                    
                    marker.pose.orientation.z=math.sin(radian/2.0)
                    marker.pose.orientation.w=math.cos(radian/2.0)
                    marker.scale.x=0.15
                    marker.scale.y=0.03
                    marker.scale.z=0.01
                    marker.color.a=1.0
                elif arrow==0:#点で表示する場合
                    marker.type=Marker.SPHERE
                    marker.scale.x=0.05
                    marker.scale.y=0.05
                    marker.scale.z=0.05
                    marker.color.a=1.0
                marker.header.frame_id='map'
                marker.header.stamp=rospy.get_rostime()
                marker.id=id
                id +=1
                marker.action=Marker.ADD
                marker.pose.position.x=sample[c][i][0]
                marker.pose.position.y=sample[c][i][1]
                marker.color.r = COLOR[c][0]
                marker.color.g = COLOR[c][1]
                marker.color.b = COLOR[c][2]

                if Number != None:
                    if Number==c:
                        marker_array.markers.append(marker)
                else:
                    if (c in no_data_class)==False:
                        marker_array.markers.append(marker)
        if mu_draw==1:
            mu_marker =Marker()
            mu_marker.type=Marker.ARROW
            mu_marker.header.frame_id='map'
            mu_marker.header.stamp=rospy.get_rostime()
            mu_marker.id=id
            id +=1  
            mu_marker.action=Marker.ADD
            mu_marker.pose.position.x=mu_all[c][0]
            mu_marker.pose.position.y=mu_all[c][1]
            #print c,mu_marker.pose.position.x,mu_marker.pose.position.y
            orient_cos=mu_all[c][3]
            orient_sin=mu_all[c][2]
            if orient_sin>1.0:
                orient_sin=1.0
            elif orient_sin<-1.0:
                orient_sin=-1.0
            #radian xを導出
            radian=math.asin(orient_sin)
            if orient_sin>0 and orient_cos<0:
                radian=radian+RAD_90
            elif orient_sin<0 and orient_cos<0:
                radian=radian-RAD_90
            
            mu_marker.pose.orientation.z=math.sin(radian/2.0)
            mu_marker.pose.orientation.w=math.cos(radian/2.0)
            #<<<<<<<矢印の大きさ変更>>>>>>>>>>>>>>>>>>>>>>>>
            mu_marker.scale.x=0.4
            mu_marker.scale.y=0.1
            mu_marker.scale.z=1.0
            mu_marker.color.a=1.0    
            if color_all==1:
                mu_marker.color.r = COLOR[c][0]
                mu_marker.color.g = COLOR[c][1]
                mu_marker.color.b = COLOR[c][2]
            elif color_all==0:
                mu_marker.color.r = 1.0
                mu_marker.color.g = 0
                mu_marker.color.b = 0
            if Number != None:
                if Number==c:
                    marker_array.markers.append(mu_marker)
            else:
                if (c in no_data_class)==False:
                    marker_array.markers.append(mu_marker)   

    print marker_array.markers
    
    while not rospy.is_shutdown():
        
        #pub.publish(marker)
        pub.publish(marker_array)
        rate.sleep()

if __name__ == '__main__':
    try:
        place_draw()
    except rospy.ROSInterruptException:
        pass
