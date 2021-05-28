#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry

class Odom_saver():

    def __init__(self):
        rospy.init_node("saver")
        self.r = rospy.Rate(50)  # 50hz
        self.save_path_as_csv = True
        self.sim_odom = Odometry()

        self.sub= rospy.Subscriber("/hsrb/odom", Odometry, self.callback)

        if self.save_path_as_csv == True:
            self.path_dict = {}


    def callback(self, message):
        #print(type(message))
        #print(type(self.sim_odom))
        rospy.loginfo("odom : x %lf  : y %lf\n", message.pose.pose.position.x, message.pose.pose.position.y)

        if self.save_path_as_csv == True:
            addRow = [0, self.sim_odom.position.x, self.sim_odom.position.y, 0, 
                     updated_quaternion[0], updated_quaternion[1], updated_quaternion[2], updated_quaternion[3],
                     self.sim_twist.linear.x, self.sim_twist.linear.y, self.sim_twist.linear.z, 0, 0, self.sim_twist.angular.z]
            self.path_dict[len(self.path_dict)] = addRow


    def odom_save(self):




    def save_csv(self):
        # Save CSV path file
        cols = ["time", "x", "y", "z", "w0", "w1", "w2", "w3", "vx", "vy", "vz", "roll", "pitch", "yaw"]
        df = pd.DataFrame.from_dict(self.path_dict, orient='index',columns=cols)
        df.to_csv("path_data.csv", index=False)


if __name__ == '__main__':
    print('Odom Saver is Started...')
    test = Odom_saver()
    
    while True:
        # 何かキーワードが押されるまでodomを保存し続ける
    
    test.save_csv()
    print("finish")
    pass