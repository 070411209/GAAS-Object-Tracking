# KCFcpp

## Dependencies
1. OpenCV
2. ROS

## Usage
Inport necessary lib
```c++
#include <ros/ros.h>
#include "RosTLD.h"
```
Init ROS node and RosTLD
```c++
int main(int argc, char** argv)
{
    ros::init(argc, argv, "ros_tld");
    RosTLD rosTLD;  //Init RosTLD.
    ros::spin();
    return 0;
}
```
Now that we have started the ROS tracking server, then set the initial rect via the ROS service.
```c++
#include "ros/ros.h"
#include "ros_tld/InitRect.h"
#include <iostream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h> 
#include <sensor_msgs/image_encodings.h> 
#include "std_msgs/Int32MultiArray.h"
#include <vector>
using namespace std;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_ros_kcf");

  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<ros_kcf::InitRect>("init_rect");
  ros_kcf::InitRect srv;
  //Initial Rect: (142,125), (232,164)
  srv.request.xmin = 142;
  srv.request.ymin = 125;
  srv.request.xmax = 232;
  srv.request.ymax = 164;
  client.call(srv);

  return 0;
}
```
Finally, publish the video frame to the KCF ROS server through ROS Publisher, and get tracking results via ROS Subscriber.
```c++
//Track from video
  VideoCapture videoCapture;
  videoCapture.open("/path/to/video");
  ros::Publisher kcf_pub = n.advertise<std_msgs::String>("camera/tracking/image", 10);
  ros::Subscriber result_sub = n.subscribe("track_rect_pub", 10, trackResultCallback);
  ros::Rate loop_rate(10);
  Mat frame;
  bool tag = videoCapture.read(frame);
  while(tag == 1)
  {
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    result_sub.publish(msg);
    tag = videoCapture.read(frame);
    ros::spinOnce();
    loop_rate.sleep();
  }
```
The above code is included in `src/testRosService.cpp`.

## Build
```sh
mkdir build
cd build
cmake ..
make -j4
```


## REFRENCE
https://blog.csdn.net/dujuancao11/article/details/109325218
https://blog.csdn.net/ljsant/article/details/75246426
https://blog.csdn.net/ljsant/article/details/75245365
https://blog.csdn.net/chenjiazhou12/article/details/29378087



---
Thanks for [Project KCFcpp](https://github.com/joaofaro/KCFcpp.git).