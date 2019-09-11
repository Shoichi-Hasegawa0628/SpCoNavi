#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion\
, Twist,PoseStamped

def callback_minic(data):
    # rospy.loginfo(data)
    pub = rospy.Publisher("/hsrb/opt_command_velocity",Twist,queue_size=10)
    twist = Twist()

    twist.linear.x = data.linear.x
    twist.linear.y = data.linear.y
    twist.linear.z = data.linear.z
    twist.angular.x = data.angular.x
    twist.angular.y = data.angular.y
    twist.angular.z = data.angular.z

    rospy.loginfo(twist)
    pub.publish(twist)
def pass_message():
    rospy.init_node("twist",anonymous=False)
    rospy.Subscriber("/cmd_vel",Twist,callback_minic)
    rospy.spin()
if __name__ == '__main__':
    # pub = rospy.Publisher("/hsrb/opt_command_velocity",Twist,queue_size=10)
    pass_message()
