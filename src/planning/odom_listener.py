#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry

class Odom_listener():

    def __init__(self):
        rospy.init_node("listener")
        self.r = rospy.Rate(50)  # 50hz
        self.sim_odom = Odometry()

        self.sub= rospy.Subscriber("/hsrb/odom", Odometry, self.callback)


    def callback(self, message):
        #print(type(message))
        #print(type(self.sim_odom))
        rospy.loginfo("Odom : x %lf  : y %lf  : z %lf\n" , message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z) #Position
        rospy.loginfo("Quaternion : w0 %lf  : w1 %lf  : w2 %lf  : w3 %lf\n" , message.pose.pose.position.x, message.pose.pose.position.y) #Quaternion
        rospy.loginfo("odom : x %lf  : y %lf\n" , message.pose.pose.position.x, message.pose.pose.position.y) #Velocity
        rospy.loginfo("odom : x %lf  : y %lf\n" , message.pose.pose.position.x, message.pose.pose.position.y) #Angular Velocity
        


if __name__ == '__main__':
    print('Odom Subscriber is Started...')
    test = Odom_listener()
    rospy.spin()

