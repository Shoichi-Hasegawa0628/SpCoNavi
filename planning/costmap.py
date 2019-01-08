#coding:utf-8
#Akira Taniguchi 2018/12/13-
#コストマップを読み込む⇒ファイル書き込み
#ROS対応：コストマップのトピックを受け取り、ファイル書き込み

import sys
import rospy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from __init__ import *
from submodules import *

#コストマップを取得⇒2次元配列の形でファイル保存
def GetCostMap(outputfile):
    #outputfolder + trialname + navigation_folder + contmap.csv
    costmap = []
    #return costmap
    



########################################
if __name__ == '__main__':
    #rospy.init_node('GetCostMap', anonymous=True)
    
    #rospy.spin()

    #学習済みパラメータフォルダ名を要求
    trialname = sys.argv[1]
    #print trialname
    #trialname = raw_input("trialname?(folder) >")
    

    outputfile = outputfolder + trialname + navigation_folder
    Makedir( outputfile )

    GetCostMap(outputfile)
    print "Get costmap"
    
