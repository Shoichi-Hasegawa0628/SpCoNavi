#!/usr/bin/env python
#coding:utf-8
# ゴール座標をServiceで行うコード
import numpy as np
import actionlib
import tf
import time
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import tf
from geometry_msgs.msg import Quaternion, Vector3
#filename = "/root/HSR/catkin_ws/src/spconavi_ros/src/data/3LDK_01/navi/path_data.csv"
filename = "/root/RULO/catkin_ws/src/spconavi_ros/src/data/3LDK_01/navi/T200N6A1S0G3_Path_ROS200.csv"

class Simple_path_simulator():

    def __init__(self):
        self.euler = Vector3()
        rospy.init_node('Simple_Path_Publisher')
        listener = tf.TransformListener()
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()
        listener.waitForTransform("map", "base_link", rospy.Time(), rospy.Duration(4.0))

        rospy.Subscriber("/odom", Odometry, self.callback)
        self.msg = [[0 for j in range(2)] for i in range(1)] #1行2列の配列の初期化
        #self.msg = PoseStamped()

        self.csv_path_data = np.loadtxt(filename, delimiter=",")
        for idx in range (len(self.csv_path_data)):
            self.idx = idx
            pose_list = self.get_poses_from_csvdata(self.idx)
            time.sleep(1.0)
            self.goal = pose_list
            time.sleep(1.0)
            client.send_goal(self.goal)
            succeeded = client.wait_for_result(rospy.Duration(5.0))
            print(succeeded)

            #    if succeeded == True:
            #        break

            #state = client.get_state()
            #if succeeded:
            #    rospy.loginfo(
            #        "Succeeded: No." + str(i + 1) + "(" + str(state) + ")")
            #else:
            #    rospy.loginfo(
            #        "Failed: No." + str(i + 1) + "(" + str(state) + ")")

            #print("goal = ",goal)
            #while True:
                #print("self.msg_x = ", self.msg[0][0], "csv_data_x = ", self.csv_path_data[self.idx][1])
                #print("self.msg_y = ", self.msg[0][1], "csv_data_y = ", self.csv_path_data[self.idx][2])
            ##    if self.msg.pose.position.x == self.csv_path_data[self.idx][1] and self.msg.pose.position.x == self.csv_path_data[self.idx][2]:
             #       print("self.msg_x = ", self.msg.pose.position.x, "csv_data_x = ", self.csv_path_data[self.idx][1])
             #       print("self.msg_y = ", self.msg.pose.position.y, "csv_data_y = ", self.csv_path_data[self.idx][2])
             #       break


    def callback(self, message):
        self.msg[0][0] = message.pose.pose.position.x
        self.msg[0][1] = message.pose.pose.position.y

    #    self.msg.pose.position.x = message.pose.pose.position.x
    #    self.msg.pose.position.y = message.pose.pose.position.y

    #def publish_path_topic(self, goal):
    #    self.odom_pub.publish(goal)
        
        
        """
        else:    
            while True:
                print("csv_path_data.X =", message.pose.pose.position.x, "real_data.X =", self.csv_path_data[self.idx][1])
                #print("csv_path_data.Y =", message.pose.pose.position.y, "real_data.Y =", self.csv_path_data[self.idx][2])
                if message.pose.pose.position.x == self.csv_path_data[self.idx][1] and message.pose.pose.position.y == self.csv_path_data[self.idx][2]:
                    break

        
        pose_list = self.get_poses_from_csvdata()
        goal =pose_list
        client.send_goal(goal)
        """
        
        #initialize publisher
        #self.path_pub = rospy.Publisher("/omni_path_follower/path", Path, queue_size=50)
        #self.path_pub = rospy.Publisher("/move_base/DWAPlannerROS/global_plan", Path, queue_size=50)

    """
    def get_poses_from_csvdata(self, idx):
        pose_list = PoseStamped()
        #print(pose_list)
        pose_list.header.frame_id = 'map'
        #pose_list = []
        pose_list.pose.position.x = self.csv_path_data[idx][1]
        pose_list.pose.position.y = self.csv_path_data[idx][2]
        pose_list.pose.position.z = 0.0
        pose_list.pose.orientation.x = 0.0
        pose_list.pose.orientation.y = 0.0
        pose_list.pose.orientation.z = 0.0
        pose_list.pose.orientation.w = 1.0

        

    """    
    def get_poses_from_csvdata(self, idx):
        #Get poses from csv data
        pose_list = MoveBaseGoal()
        pose_list.target_pose.header.frame_id = 'map'
        pose_list.target_pose.pose.position.x = self.csv_path_data[idx][1]
        pose_list.target_pose.pose.position.y = self.csv_path_data[idx][0]
        pose_list.target_pose.pose.position.z = 0.0
        if idx == 0:
            print("pose_list:", pose_list)

        if idx == 0:
            self.euler.x = 0
            self.euler.y = 0
            #angular = math.atan((self.csv_path_data[idx][0]-self.msg[0][1])/(self.csv_path_data[idx][1]-self.msg[0][0]))
            angular = math.atan((self.msg[0][1]-self.csv_path_data[idx][0])/(self.msg[0][0]-self.csv_path_data[idx][1]))
            self.euler.z = math.radians(angular)
            quaternion = self.euler_to_quaternion (self.euler)

        if idx > 0:
            self.euler.x = 0
            self.euler.y = 0
            angular = math.atan((self.csv_path_data[idx][0]-self.csv_path_data[idx-1][0])/(self.csv_path_data[idx][1]-self.csv_path_data[idx-1][1]))
            self.euler.z = math.radians(angular)
            quaternion = self.euler_to_quaternion (self.euler)

        pose_list.target_pose.pose.orientation.x = quaternion.x
        pose_list.target_pose.pose.orientation.y = quaternion.y
        pose_list.target_pose.pose.orientation.z = quaternion.z
        pose_list.target_pose.pose.orientation.w = quaternion.w
    
        return pose_list

    def euler_to_quaternion(self, euler):
        """Convert Euler Angles to Quaternion

        euler: geometry_msgs/Vector3
        quaternion: geometry_msgs/Quaternion
        """
        q = tf.transformations.quaternion_from_euler(euler.x, euler.y, euler.z)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    """
        #poses_list = []
        #print(self.csv_path_data)
        #for indx in range(len(self.csv_path_data)):
        #print(indx)

        temp_pose.pose.position.x = self.csv_path_data[indx][1]
        temp_pose.pose.position.y = self.csv_path_data[indx][2]
        temp_pose.pose.position.z = self.csv_path_data[indx][3]
        temp_pose.pose.orientation.x = self.csv_path_data[indx][4]
        temp_pose.pose.orientation.y = self.csv_path_data[indx][5]
        temp_pose.pose.orientation.z = self.csv_path_data[indx][6]
        temp_pose.pose.orientation.w = self.csv_path_data[indx][7]
        temp_pose.header = self.path_header
        temp_pose.header.seq = indx
        poses_list.append(temp_pose)
    """

        


if __name__ == '__main__':
    print('Path Publisher is Started...')
    test = Simple_path_simulator()
    
    try:
        while not rospy.is_shutdown():
            test.publish_path_topic(test.goal)
            print('Success!!!!')
    except KeyboardInterrupt:
        print("finished!")
    #rospy.spin()

