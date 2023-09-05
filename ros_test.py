
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped



def listener_wait_msg():

    rospy.init_node('ros_subscription_test_node')

    cube_msg = rospy.wait_for_message('optitrack/cube_rainbow/poseStamped', PoseStamped)

    return cube_msg.pose.position

if __name__ == '__main__':

    msg_count = 0

    while True:
 
        cube_pos = listener_wait_msg()
        cube_pos_array = np.array([cube_pos.x, cube_pos.y]) -np.array([ 0.20667699, -0.02906933]) +np.array([0, -0.3])
  

        msg_count += 1
        print("ctrl : ", cube_pos_array)
 
        #print("rel vec : ", ctrl_pos_array-ref_pos_array)

        if msg_count >= 10000: break

    print('Done!')
