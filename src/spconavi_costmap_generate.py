#!/usr/bin/env python
#coding:utf-8

# 標準ライブラリ
import sys

# サードパーティー
import numpy as np
import rospy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid

# 自作ライブラリ
from __init__ import *
from spconavi_math import *

class CostMap(object):

    def do_mkdir(self):
        # Request folder name of the learned model parameters (In the SIGVerse experiment, it was room name ID.)
        self.trialname = sys.argv[1]
        #print trialname
        #trialname = raw_input("trialname?(folder) >")
        
        self.trialfolder = outputfolder_SIG + self.trialname
        Makedir( self.trialfolder )
        print "make dir:", self.trialfolder
        
        self.outputfile = self.trialfolder + costmap_folder
        Makedir( self.outputfile )
        print "make dir:", self.outputfile

    def map_callback(self, hoge):

        self.map = hoge

        self.MapData = np.array([self.map.data[i:i+self.map.info.width] for i in range(0, len(self.map.data), self.map.info.width)])
        print "get map data."
        
        # Save file
        np.savetxt(self.outputfile + "map.csv", self.MapData, delimiter=",")
        print "save map."
        print self.outputfile + "map.csv"

    def costmap_callback(self, hoge):

        self.costmap = hoge

        self.CostmapData = np.array([self.costmap.data[i:i+self.costmap.info.width] for i in range(0, len(self.costmap.data), self.costmap.info.width)])
        print "get costmap data."
        
        # Save file
        np.savetxt(self.outputfile + "costmap.csv", self.CostmapData, delimiter=",")
        print "save costmap."
        print self.outputfile + "costmap.csv"
    
    def __init__(self):

        self.do_mkdir()
        rospy.Subscriber(MAP_TOPIC, OccupancyGrid, self.map_callback, queue_size=1)
        print "map ok"
        rospy.Subscriber(COSTMAP_TOPIC, OccupancyGrid, self.costmap_callback, queue_size=1)
        print "costmap ok"



if __name__ == '__main__':
    print "Ctrl-C is the end of process."
    rospy.init_node('CostMap', anonymous=True)
    hoge = CostMap()
    rospy.spin()

    print "\n [Done] Get map and costmap."
    
