#ifndef __RosKCF__
#define __RosKCF__

#include <ros/ros.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/image_encodings.h> 
#include <image_transport/image_transport.h>
#include "std_msgs/Int32MultiArray.h"
#include "ros_kcf/InitRect.h"
#include "kcftracker.hpp"


class RosKCF
{
private:
    cv::Rect *initRectPtr;
    std_msgs::Int32MultiArray trackResultMsg;

    ros::NodeHandle nodeHandle;
    ros::Publisher trackPub;
    ros::ServiceServer service;
    image_transport::Subscriber imageSub;
    image_transport::Subscriber dispritySub;
    KCFTracker *kcfPtr;

    bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool LAB = false;
    std::string _rgb_topic;
    std::string _dsp_topic;
    cv::Mat disp_img_;
    cv::Mat depth_img_;
    cv::Rect result_;
	cv::Mat hist_;
    cv::Mat hist_n_;   

    std::string output = "Debug";

public:

    RosKCF();
    //callback of service server.
    bool setInitRect(ros_kcf::InitRect::Request &req, ros_kcf::InitRect::Response &res);

    //callback of imageSub.
    void buildAndTrack(const sensor_msgs::ImageConstPtr &msg);

    void computerTrackPose(const sensor_msgs::ImageConstPtr &msg);
    
    void computerDepthRegion();    
    
    void getTrackShow(cv::Mat _img, cv::Rect _rect);

    void drawHistGraph(cv::Mat _img);

    void accValue(cv::Mat _img);
};

#endif