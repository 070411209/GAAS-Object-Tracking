#include "RosKCF.h"
#include <ros/ros.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h> 
#include <sensor_msgs/image_encodings.h> 
#include "ros_kcf/InitRect.h"
#include "std_msgs/Int32MultiArray.h"
#include <vector>
#include "kcftracker.hpp"

using namespace std;
using namespace cv;

RosKCF::RosKCF():_rgb_topic("/realsense/color/image_raw"), _dsp_topic("/realsense/depth/image_rect_raw")
{
    this->service = nodeHandle.advertiseService("init_rect", &RosKCF::setInitRect, this);
    this->trackPub = nodeHandle.advertise<std_msgs::Int32MultiArray>("track_rect_pub", 1000);
    image_transport::ImageTransport it(nodeHandle);
    this->imageSub = it.subscribe(_rgb_topic, 100, &RosKCF::buildAndTrack, this);
    this->dispritySub = it.subscribe(_dsp_topic, 100, &RosKCF::computerTrackPose, this);
    this->initRectPtr = NULL;
    this->kcfPtr = NULL;
}

bool RosKCF::setInitRect(ros_kcf::InitRect::Request &req, ros_kcf::InitRect::Response &res)
{
    int tlx = req.xmin;
    int tly = req.ymin;
    int brx = req.xmax;
    int bry = req.ymax;
    this->initRectPtr = new cv::Rect(Point2d(tlx, tly), Point2d(brx, bry));
    std::cout<<"Init rect received."<<std::endl;
    return true;
}

void RosKCF::buildAndTrack(const sensor_msgs::ImageConstPtr &msg)
{
    try{
        cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
        if (initRectPtr == NULL) return;

        if (kcfPtr == NULL && initRectPtr != NULL)
        {
            cout<<"Build"<<endl;
            kcfPtr = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
            kcfPtr->init(*initRectPtr, img);

            return ;
        }
        else
        {
            
            result_ = kcfPtr->update(img);

            int x1 = result_.tl().x;
            int y1 = result_.tl().y;
            int x2 = result_.br().x;
            int y2 = result_.br().y;
            cout<<"Track rect: {["<< x1 << " " << y1 << "], ["<< x2 << " "<< y2 <<"]}"<<endl;

            vector<int> trackResult;
            trackResult.push_back(result_.tl().x);
            trackResult.push_back(result_.tl().y);
            trackResult.push_back(result_.br().x);
            trackResult.push_back(result_.br().y);
            trackResultMsg.data = trackResult;
            trackPub.publish(trackResultMsg);

            getTrackShow(img, result_);
            return ;
        }
        
    }catch(cv_bridge::Exception &e){
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}


void RosKCF::computerTrackPose(const sensor_msgs::ImageConstPtr &msg)
{
    try{
        depth_img_ = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;     
        computerDepthRegion();
    }catch(cv_bridge::Exception &e){
        ROS_ERROR("Could not convert from '%s' to 'TYPE_32FC1'.", msg->encoding.c_str());
    }
}

void RosKCF::computerDepthRegion()
{
    cv::Mat roi_depth = depth_img_(result_);
    // drawHistGraph(roi_depth); 
    accValue(roi_depth);
#if 0
    std::cout << "ROI:" << roi_depth.cols << " " << roi_depth.rows << std::endl;    
#endif    
}

void RosKCF::drawHistGraph(cv::Mat _img)
{
    int histSize[1] = {200};    // 指的是直方图分成多少个区间
    int channels[1] = {0};
    float hr1[] = {0, 20};      //矩阵b的深度值有效范围是0<=V<20
    const float* ranges[] = {hr1};    
    int hist_h = 200;       //直方图的图像的高
    int hist_w = 400;       //直方图的图像的宽    
    int bin_w = hist_w / histSize[0];  //直方图的等级
    float max_hist = 0.0;
    Point max_point(0.0, 0.0);

    cv::Mat histImage = cv::Mat(hist_h, hist_w, CV_8UC1, Scalar(0));//绘制直方图显示的图像
    calcHist(&_img, 1, channels, Mat(), hist_, 1, histSize, ranges, true);
    normalize(hist_, hist_, 0, hist_h, cv::NORM_MINMAX, -1, Mat());   //归一化
	for (int i = 1; i < histSize[0]; i++)
	{
        if(hist_.at<float>(i - 1) > max_hist) {
            max_hist = hist_.at<float>(i - 1);
            max_point.x = float(i)/10;
            max_point.y = max_hist;
        }

		line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(hist_.at<float>(i - 1))),
			Point((i)*bin_w, hist_h - cvRound(hist_.at<float>(i))), Scalar(100), 1, CV_AA);
#if 0
		line(histImage, Point((i - 1)*bin_w, hist_h),  
            Point((i - 1)*bin_w, hist_h - cvRound(hist_.at<float>(i - 1))), Scalar(100), 1);      
#endif              
	}

    std::cout << "Max Point : " << max_point << std::endl;
	cv::imshow(output, histImage);
    cv::waitKey(10);
}

void RosKCF::accValue(cv::Mat _img)
{
    int ocp_pixels = 0;
    float accm_value = 0.0f;    
    auto pixels_num = result_.area();
    float avg_depth = 0.0f;

    std::vector<float> ocp_depths(pixels_num);
    if (!_img.empty()) {
        for (int x = 0; x < _img.cols; x++) {
            for (int y = 0; y < _img.rows; y++) {
                float d = _img.at<float>(y, x);
                if (d > 0.1f) {
                    ocp_depths[ocp_pixels] = d;
                    accm_value += d;
                    ocp_pixels++;
                }
            }
        }
    }
    if(ocp_pixels > 0)
        avg_depth = accm_value / ocp_pixels;
#if 0    
    std::cout << "AVG_DEPTH : " << avg_depth << ", ocp_pixels : " << ocp_pixels << std::endl;
#endif
}

void RosKCF::getTrackShow(cv::Mat _img, cv::Rect _rect)
{
    cv::rectangle(_img, _rect, Scalar(255, 0, 0), 1, 1, 0);
    cv::imshow("TrackerShow", _img);
    cv::waitKey(10);
}
