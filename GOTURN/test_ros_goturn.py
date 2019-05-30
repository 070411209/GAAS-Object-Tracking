from __future__ import print_function
import rospy
#Add ROS service to python system path.
import sys
sys.path.append('/path/to/GOTURN/build/devel/lib/python2.7/dist-packages')
from goturn_ros.srv import InitRect
import os
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from goturn.ros import config

f = open('result.txt', 'a')

def get_rect_result(result):
    print("xmin:{}, ymin:{}, xmax:{}, ymax:{}".format(result.data[0],result.data[1],result.data[2],result.data[3]))
    f.write("{},{},{},{}\n".format(result.data[0],result.data[1],result.data[2],result.data[3]))

if __name__ == "__main__":
    data_dir = "/path/to/image/dir"
    rospy.wait_for_service('init_rect')
    init_rect = rospy.ServiceProxy('init_rect', InitRect)
    try:
        f.write('142,125,232,164\n')
        init_rect(142,125,232,164)  #142,125,232,164 from TLD Dataset 06_car.
    except:
        pass

    bridge = CvBridge()
    rospy.init_node('test_ros_goturn', anonymous=True)
    img_pub = rospy.Publisher(config.IMAGE_SUB_TOPIC, Image, queue_size=10)
    result_sub = rospy.Subscriber(config.TRACK_PUB_TOPIC, Int32MultiArray, get_rect_result)
    rate = rospy.Rate(10)
    file_list = os.listdir(data_dir)
    file_list.sort(key = lambda name: int(name.split('.')[0]))
    
    for name in file_list:
        if name.endswith('.jpg'):
            img_path = os.path.join(data_dir, name)
            img = cv2.imread(img_path)
            msg = bridge.cv2_to_imgmsg(img, "bgr8")
            img_pub.publish(msg)
            rate.sleep()

