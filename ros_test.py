
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


def callback(data):
    rospy.loginfo(f"{rospy.get_caller_id()}\n{data.pose.position}")
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('ros_subscription_test_node')
    rospy.Subscriber('optitrack/test_bj2/poseStamped', PoseStamped, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    return None

def listener_wait_msg():

    rospy.init_node('ros_subscription_test_node')

    msg = rospy.wait_for_message('optitrack/ctrl_jh/poseStamped', PoseStamped)

    return msg.pose.position

if __name__ == '__main__':

    msg_count = 0

    while True:
 
        pos = listener_wait_msg()

        msg_count += 1

        if (msg_count % 10) == 0:
            print(f'x: {pos.x:6.3f} | y: {pos.y:6.3f} | z: {pos.z:6.3f}')

        if msg_count >= 10000: break

    print('Done!')
