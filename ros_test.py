
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


def listener_wait_msg():

    rospy.init_node('ros_subscription_test_node')

    cube_msg = rospy.wait_for_message('optitrack/cube_jh/poseStamped', PoseStamped)
    #ref_msg = rospy.wait_for_message('optitrack/ref_jh/poseStamped', PoseStamped)

    return cube_msg.pose.position

if __name__ == '__main__':

    msg_count = 0

    while True:
 
        cube_pos = listener_wait_msg()
        
        cube_pos_array = np.array([cube_pos.x, cube_pos.y])


        msg_count += 1
        print("ctrl : ", cube_pos_array- [0.11719225 ,2.44359732] + [0, -0.4])
 
        #print("rel vec : ", ctrl_pos_array-ref_pos_array)

        if msg_count >= 10000: break

    print('Done!')
