#!/usr/bin/env python
#coding:utf-8
# 計算したデータを読み込み, 「ロボット(HSR)にある軌道を沿うように命令」と「計算したPathを可視化」させるプログラム

# サードパーティー
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path

filename = "/root/HSR/catkin_ws/src/spconavi_ros/data/3LDK_01/navi/Astar_Approx_expect_N6A1S(192, 192)G3_Path_ROS.csv"

class Simple_path_simulator():

    def __init__(self):
        rospy.init_node('Simple_Path_Publisher')
        self.path_pub = rospy.Publisher("/omni_path_follower/path", Path, queue_size=50) 
        #self.path_pub = rospy.Publisher("/spconavi_plan", Path, queue_size=50)
        self.r = rospy.Rate(50)  
        self.path_header = Header()
        self.path_header.seq = 0
        self.path_header.stamp = rospy.Time.now()
        self.path_header.frame_id = "map"

        self.path = Path()
        self.path.header = self.path_header

        self.csv_path_data = np.loadtxt(filename, delimiter=",")
        pose_list = self.get_poses_from_csvdata()
        self.path.poses =pose_list


    def get_poses_from_csvdata(self):
        poses_list = []
        #print(self.csv_path_data)
        for indx in range(len(self.csv_path_data)):
            temp_pose = PoseStamped()
            temp_pose.header.frame_id = 'map'
            temp_pose.pose.position.x = self.csv_path_data[indx][1]
            temp_pose.pose.position.y = self.csv_path_data[indx][0]
            poses_list.append(temp_pose)
        return poses_list

    
    def publish_path_topic(self):
        self.path_pub.publish(self.path)
        self.r.sleep()


if __name__ == '__main__':
    print('Path Publisher is Started...')
    test = Simple_path_simulator()
    try:
        while not rospy.is_shutdown():
            test.publish_path_topic()
            print('Success!!!!')
    except KeyboardInterrupt:
        print("finished!")

