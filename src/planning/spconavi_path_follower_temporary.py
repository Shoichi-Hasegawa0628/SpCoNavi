#!/usr/bin/env python
#coding:utf-8
# ある(x, y, θ)に向けてナビゲーションを行うプログラム
import numpy as np
import tf
import math
import time
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion, Vector3


class Simple_path_simulator():

    def __init__(self):
        rospy.init_node('Simple_Path_Publisher')
        self.euler = Vector3()
        self.odom_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=50)

        pose_list = PoseStamped()
        pose_list.header.frame_id = 'map'
        pose_list.pose.position.x = -1.95
        pose_list.pose.position.y = 3.3
        pose_list.pose.position.z = 0.0

        self.euler.x = 0
        self.euler.y = 0
        self.euler.z = math.radians(180)
        quaternion = self.euler_to_quaternion (self.euler)

        pose_list.pose.orientation.x = quaternion.x
        pose_list.pose.orientation.y = quaternion.y
        pose_list.pose.orientation.z = quaternion.z
        pose_list.pose.orientation.w = quaternion.w
        
        time.sleep(1.0)
        self.goal = pose_list
        self.publish_path_topic(self.goal)
        print(self.goal)


    def euler_to_quaternion(self, euler):
        """Convert Euler Angles to Quaternion

        euler: geometry_msgs/Vector3
        quaternion: geometry_msgs/Quaternion
        """
        q = tf.transformations.quaternion_from_euler(euler.x, euler.y, euler.z)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


    def publish_path_topic(self, goal):
        self.odom_pub.publish(goal)
        

if __name__ == '__main__':
    print('Path Publisher is Started...')
    test = Simple_path_simulator()
 
    

