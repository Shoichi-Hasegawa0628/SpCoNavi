#!/usr/bin/env python
#coding:utf-8
# 計算したデータを読み込み, 「ロボット(RULO)にある(x, y, θ)地点への命令」と「計算したPathを可視化」させるプログラム

# 標準ライブラリ
import math
import time

# サードパーティー
import numpy as np
import tf
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion, Vector3
from std_msgs.msg import String, Header, Empty
from nav_msgs.msg import Path

class Simple_path_simulator():

    def __init__(self):
        rospy.init_node('Simple_Path_Publisher')
        self.euler = Vector3()
        self.pose_list = PoseStamped()
        self.r = rospy.Rate(10) 
        self.odom_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=50) #目的地の配信
        self.path_pub = rospy.Publisher("/spconavi_plan", Path, queue_size=50) # rvizで可視化用のPath配信
        self.next_pub = rospy.Publisher("/next_judge", Empty, queue_size=50) # 判別プログラム以降用のメッセージ配信
        self.path_header = Header()
        self.path_header.seq = 0
        self.path_header.stamp = rospy.Time.now()
        self.path_header.frame_id = "map"
        self.path = Path()
        self.path.header = self.path_header
        self.next_judge = Empty()
        self.main()


    def main(self):
        # データの読み込み
        filepath = rospy.wait_for_message("/next_state", String, timeout=None)
        csv_path_data = np.loadtxt(filepath.data, delimiter=",")
        #filepath = "/root/RULO/catkin_ws/src/spconavi_ros/data/3LDK_01/navi/Astar_Approx_expect_N6A1SX192Y192G1_Path_ROS.csv"
        #csv_path_data = np.loadtxt(filepath, delimiter=",")

        # 目的地の地点の設定
        idx = len(csv_path_data) - 1
        pose_list = self.get_destination_from_csvdata(idx, csv_path_data)
        goal = pose_list
        print(goal)

        # 可視化用のPathをROSのPathメッセージに格納
        path = self.get_path_from_csvdata(csv_path_data)
        self.path.poses = path
        #print(path)

        #while not rospy.is_shutdown():
        for t in range(0, 30, 1):
            self.odom_pub.publish(goal)
            self.path_pub.publish(self.path)
            time.sleep(1)
            #self.r.sleep()

    
    def get_destination_from_csvdata(self, idx, csv_path_data):
        #print(self.pose_list)
        self.pose_list.header.frame_id = 'map'
        self.pose_list.pose.position.x = csv_path_data[idx][1]
        self.pose_list.pose.position.y = csv_path_data[idx][0]
        self.pose_list.pose.position.z = 0.0

        self.euler.x = 0
        self.euler.y = 0
        self.euler.z = math.radians(270) #0, 90, 180, 270 (場所によって姿勢を変えたほうがいいかも？)
        quaternion = self.euler_to_quaternion (self.euler)
        self.pose_list.pose.orientation.x = quaternion.x
        self.pose_list.pose.orientation.y = quaternion.y
        self.pose_list.pose.orientation.z = quaternion.z
        self.pose_list.pose.orientation.w = quaternion.w
        return self.pose_list


    def get_path_from_csvdata(self, csv_path_data):
        path = []
        #print(csv_path_data)
        for idx in range(len(csv_path_data)):
            temp_pose = PoseStamped()
            temp_pose.header.frame_id = 'map'
            temp_pose.pose.position.x = csv_path_data[idx][1]
            temp_pose.pose.position.y = csv_path_data[idx][0]
            path.append(temp_pose)
        return path


    def euler_to_quaternion(self, euler):
        q = tf.transformations.quaternion_from_euler(euler.x, euler.y, euler.z)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        

if __name__ == '__main__':
    print('Path Publisher is Started...')
    test = Simple_path_simulator()
    for i in range (0, 10, 1):
        test.next_pub.publish(test.next_judge)
        time.sleep(1)
    

