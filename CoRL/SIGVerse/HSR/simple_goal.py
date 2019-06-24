#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion\
, Twist,PoseStamped

import time
def Goal():
    pub = rospy.Publisher("/move_base_simple/goal",PoseStamped,queue_size=10)

    rospy.init_node("MyGoal_Coordinate",anonymous=False)

    r = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     goal = PoseStamped()
    #     goal.header.frame_id="odom"
    #     goal.pose.position.x = 3.0188
    #     goal.pose.position.y = 3.8347
    #     goal.pose.position.z = 0.0
    #     goal.pose.orientation.x = 0.0
    #     goal.pose.orientation.y = 0.0
    #     goal.pose.orientation.z = 0.4065
    #     goal.pose.orientation.w = 0.9136
    #
    #     pub.publish(goal)
    #     r.sleep()
    goal = PoseStamped()
    # goal.header.frame_id="odom"
    # goal.pose.position.x = 3.0188
    # goal.pose.position.y = 3.8347
    # goal.pose.position.z = 0.0
    # goal.pose.orientation.x = 0.0
    # goal.pose.orientation.y = 0.0
    # goal.pose.orientation.z = 0.4065
    # goal.pose.orientation.w = 0.9136


    goal.header.frame_id="odom"
    goal.pose.position.x = 6.3582
    goal.pose.position.y = -5.746
    goal.pose.position.z = 0.0
    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = -0.5916
    goal.pose.orientation.w = 0.8062

    rospy.loginfo(goal)

    time.sleep(10)
    pub.publish(goal)



if __name__ == '__main__':
    try:
        Goal()
    except rospy.ROSInterruptException:pass
