#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
import PyKDL
sys.path.append("../lib/")
import file_read as f_r

RAD_90=math.radians(90)
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
[0.3,0,0.7],[0,0.7,0.3],[0,0.3,0.7],[0.25,0.25,0.5],[0.25,0.5,0.25], #99
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
Number=int(sys.argv[2])
region_count=np.loadtxt(Parameter_diric+"/region_count.txt")
print 79
list_num=[84,4,140,116]
#for i in range(len(region_count)):
#    if region_count[i]>0:
#        list_num.append(i)
def place_draw():
    id=0
    pub = rospy.Publisher('draw_space',MarkerArray, queue_size = 10)
    rospy.init_node('draw_spatial_concepts', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    marker_array=MarkerArray()
    mu_all=f_r.mu_read(Parameter_diric)
    sigma=f_r.sigma_read(Parameter_diric)
    for i in list_num:
        #c=Number
        c=Number
        id+=1

        marker =Marker()
        marker.type=Marker.CYLINDER
                
        (eigValues,eigVectors) = np.linalg.eig(sigma[c])
        angle = (math.atan2(eigVectors[1, 0], eigVectors[0, 0]));
                
        marker.scale.x = 2*math.sqrt(eigValues[0])*3;
        marker.scale.y = 2*math.sqrt(eigValues[1])*3;
                
        marker.pose.orientation.w = math.cos(angle*0.5);
        marker.pose.orientation.z = math.sin(angle*0.5);
                
                
        marker.scale.z=0.01 # default: 0.05
        marker.color.a=0.3
        marker.header.frame_id='map'
        marker.header.stamp=rospy.get_rostime()
        marker.id=id
        id +=1
        marker.action=Marker.ADD
        marker.pose.position.x=mu_all[c][0]
        marker.pose.position.y=mu_all[c][1]
        marker.color.r = COLOR[c][0] # default: COLOR[c][0] 色のばらつきを広げる
        marker.color.g = COLOR[c][1] # default: COLOR[c][1] 色のばらつきを広げる
        marker.color.b = COLOR[c][2] # default: COLOR[c][2] 色のばらつきを広げる
        marker_array.markers.append(marker)


        mu_marker =Marker()
        mu_marker.type=Marker.ARROW
        mu_marker.id=id
        mu_marker.header.frame_id='map'
        mu_marker.header.stamp=rospy.get_rostime()

        mu_marker.action=Marker.ADD
        mu_marker.pose.position.x=mu_all[c][0]
        mu_marker.pose.position.y=mu_all[c][1]
        mu_marker.color.r = COLOR[c][0] # default: COLOR[c][0] 色のばらつきを広げる
        mu_marker.color.g = COLOR[c][1] # default: COLOR[c][1] 色のばらつきを広げる
        marker.color.b = COLOR[c][2] 
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
        mu_marker.scale.x=0.3 # default: 0.4
        mu_marker.scale.y=0.07 # default: 0.1
        mu_marker.scale.z=0.001 # default: 1.0
        mu_marker.color.a=1.0
        marker_array.markers.append(mu_marker)
    print marker_array
    while not rospy.is_shutdown():
            
        #pub.publish(marker)
        pub.publish(marker_array)
        rate.sleep()
if __name__ == '__main__':
    try:
        place_draw()
    except rospy.ROSInterruptException:
        pass
