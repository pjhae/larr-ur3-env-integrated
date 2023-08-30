
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


def listener_wait_msg():

    rospy.init_node('ros_subscription_test_node')

    ctrl_msg = rospy.wait_for_message('optitrack/ctrl_jh/poseStamped', PoseStamped)
    ref_msg = rospy.wait_for_message('optitrack/ref_jh/poseStamped', PoseStamped)

    return ctrl_msg.pose.position, ref_msg.pose.position

if __name__ == '__main__':

    msg_count = 0

    while True:
 
        ctrl_pos, ref_pos = listener_wait_msg()
        
        ctrl_pos_array = np.array([ctrl_pos.x, ctrl_pos.y, ctrl_pos.z])
        ref_pos_array  = np.array([ref_pos.x, ref_pos.y, ref_pos.z])

        msg_count += 1
        print("ctrl : ", ctrl_pos_array)
        print("ref : ", ref_pos_array)
        print("rel vec : ", ctrl_pos_array-ref_pos_array)

        if msg_count >= 10000: break

    print('Done!')
