#! /usr/bin/env python

import glob
import math
import re
import csv
import rospy
import numpy as np
import time

from rgiro_spco2_visualization_msgs.msg import GaussianDistributions, GaussianDistribution
from rgiro_spco2_visualization_msgs.srv import GaussianService, GaussianServiceRequest
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import std_msgs.msg
import os

#LMweight = "weight" 
datafolder   = "/root/RULO/catkin_ws/src/spconavi_ros/src/data/3LDK_01"


class EmSpcotRviz(object):

    def __init__(self):

        pub = rospy.Publisher("transfer_learning/gaussian_distribution",GaussianDistributions, queue_size=10)
        rospy.sleep(0.5)
        rospy.loginfo("start visualization!!")
 
        ################################################################################################################
        #       Init Path, Load File
        ################################################################################################################
        
        
        #trialname="test"
        """
        step=0
        while os.path.exists(datafolder + trialname + '/' + str(step+1)):
            step += 1
        print "step",step

        if (LMweight != "WS"):
            omomi = '/weights.csv'
        else: #if (LMweight == "WS"):
            omomi = '/WS.csv'
        
        i = 0
        for line in open(datafolder + trialname + '/'+ str(step) + omomi, 'r'):
            if (i == 0):
                max_particle = int(line)
                i += 1
        """
 
        filename = datafolder + '/3LDK_01_Myu_1_0' + '.csv'
        mu = []
        #print"max_particle:",max_particle
        print "filename:",filename 
        mu = np.genfromtxt(filename,delimiter=',')

        print "mu",mu


        sigma = []
        filename = datafolder + '/3LDK_01_S_1_0' + '.csv'
        sigma = np.genfromtxt(filename,delimiter=',' )

        print "sigma",sigma

        #color = np.loadtxt("./color.csv", delimiter=",")
        #script_dir_abspath = os.path.dirname(os.path.abspath(__file__))
        color = np.loadtxt("/root/RULO/catkin_ws/src/rgiro_spco2_visualization/rgiro_spco2_visualization/src/color.csv", delimiter=",")

        if mu.ndim == 1:
            region_num=1
        else:
            region_num = len(sigma)
        print "num",region_num

        ################################################################################################################
        #       Publish Gaussian Distribution
        ################################################################################################################
        distributions = GaussianDistributions()
        # print()
        print("R = ", region_num)
        for i in range(region_num):

                word_index = i#word_index_dic[w]
                print "i;",i
                distribution = GaussianDistribution()
                if mu.ndim == 1:
                    distribution.mean_x = mu[0]
                    distribution.mean_y = mu[1]
                else:
                    distribution.mean_x = mu[i][0]
                    distribution.mean_y = mu[i][1]

                if sigma.ndim == 1:
                    distribution.variance_x = np.sqrt(sigma[0])
                    distribution.variance_y = np.sqrt(sigma[3])
                    distribution.covariance = sigma[1]#[0]
                    correlation_coefficient = (sigma[1])**2 / (sigma[0] * sigma[3])
                    print(np.pi)
                    print(correlation_coefficient)
                    print(sigma[0])
                    print(sigma[1])
                    print(sigma[2])
                    print(sigma[3])
                    distribution.probability = 1.0 / (2.0 * np.pi * np.sqrt(sigma[0] * sigma[3] * (1.0 - correlation_coefficient))) #word[pi_max[i]][np.argsort(word[pi_max[i]])[::-1][0]]
                else:
                    distribution.variance_x = np.sqrt(sigma[i][0])
                    distribution.variance_y = np.sqrt(sigma[i][3])
                    distribution.covariance = sigma[i][1]#[0]
                    correlation_coefficient = (sigma[i][1])**2 / (sigma[i][0] * sigma[i][3])
                    distribution.probability = 3 * 1.0 / (2.0 * np.pi * np.sqrt(sigma[i][0] * sigma[i][3] * (1.0 - correlation_coefficient))) #word[pi_max[i]][np.argsort(word[pi_max[i]])[::-1][0]]
                print(distribution.probability)
                distribution.r = int(color[word_index][0])
                distribution.g = int(color[word_index][1])
                distribution.b = int(color[word_index][2])
                distributions.distributions.append(distribution)

        rate = rospy.Rate(1) # 1 Hz

        pub.publish(distributions)
        rate.sleep()
 
#def callback(message):

#    EmSpcotRviz()


if __name__ == "__main__":
    rospy.init_node("spcot_rviz", anonymous=False)
    print "start visualization"
    #rate = rospy.Rate(1) # 1 Hz
    #rospy.Subscriber('start_visualization', String, callback)
    EmSpcotRviz()
    #rospy.Subscriber('speech_to_text', String, callback)
    rospy.spin()
