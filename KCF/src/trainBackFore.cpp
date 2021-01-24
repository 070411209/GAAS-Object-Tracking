//运动物体检测——背景减除法
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <iostream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h> 
#include <sensor_msgs/image_encodings.h> 
#include "std_msgs/Int32MultiArray.h"

#include <ros_kcf/InitRect.h>
#include<thread>  

using namespace std;
using namespace cv;

class RosDetect
{
private:
    ros::NodeHandle nodeHandle;
    ros::Publisher trackPub;
    image_transport::Subscriber imageSub;
    image_transport::Subscriber dispritySub;
    ros::Subscriber cameraInfoSub;
    ros::ServiceClient client_;

    cv::Mat frame; //frame image
    cv::Mat frImg; //foreground image
    cv::Mat bkImg; //background image

    cv::Mat frameMat; //frame Mat
    cv::Mat frMat;    //foreground image
    cv::Mat bkMat;    //foreground image
    cv::Mat result;
    cv::Mat frcontImg;
    cv::Mat tmp_frame, bgmask, out_frame;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<Vec4i> hierarchy;
    int nFrmNum = 0;
    ros_kcf::InitRect srv;
    float fBF_;
    int width_;
    int height_;
    float fBaseline_;
    float fFocus_;
    float cx_;
    float cy_;
    cv::Mat depth_img_;
    float avg_depth = 0.0f;
    float avg_sat = 0.0f;
    cv::Mat kernel_erode;
    cv::Mat kernel_dilate;
    Ptr<BackgroundSubtractor> bgsubtractor;
public:
    RosDetect();
    int start();
    void rawImageCb(const sensor_msgs::ImageConstPtr &msg);
    void disprityCb(const sensor_msgs::ImageConstPtr &msg);
    void onCameraInfoCb(const sensor_msgs::CameraInfoConstPtr &msg);  
    int detectService(cv::Rect& rect_);
    void cmpRegionDepth(cv::Mat _img);
    void refineSegments(const Mat& img, Mat& mask, Mat& dst);
};

RosDetect::RosDetect()          
{
#if 1
    //创建窗口
    namedWindow("video", 1);
    // namedWindow("background", 1);
    namedWindow("foreground", 1);
    namedWindow("result", 1);
    //使窗口有序排列
    cvMoveWindow("video", 30, 0);
    // cvMoveWindow("background", 360, 0);
    cvMoveWindow("foreground", 300, 0);
    cvMoveWindow("result", 600, 0);
#endif
    image_transport::ImageTransport it(nodeHandle);
    this->client_ = nodeHandle.serviceClient<ros_kcf::InitRect>("init_rect");
    this->imageSub = it.subscribe("/realsense/color/image_raw", 10, &RosDetect::rawImageCb, this);
    this->dispritySub = it.subscribe("/realsense/depth/image_rect_raw", 10, &RosDetect::disprityCb, this);    
    this->cameraInfoSub = nodeHandle.subscribe("/realsense/color/camera_info", 10, &RosDetect::onCameraInfoCb, this);
}

void RosDetect::rawImageCb(const sensor_msgs::ImageConstPtr &msg)
{
    try {
        cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
        img.copyTo(frame);
    } catch(cv_bridge::Exception &e){
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void RosDetect::disprityCb(const sensor_msgs::ImageConstPtr &msg)
{
    try{
        depth_img_ = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;    
    }catch(cv_bridge::Exception &e){
        ROS_ERROR("Could not convert from '%s' to 'TYPE_32FC1'.", msg->encoding.c_str());
    }
}

void RosDetect::onCameraInfoCb(const sensor_msgs::CameraInfoConstPtr &msg)
{
    width_ = msg->width;
    height_ = msg->height;
    fBaseline_ = msg->K[3];
    fFocus_ = msg->K[0];
    cx_ = msg->K[2];
    cy_ = msg->K[5];
    fBF_ = fFocus_ * fBaseline_;
}

void RosDetect::refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
	int niters = 3;
 
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
 
	Mat temp;
 
	dilate(mask, temp, Mat(), Point(-1,-1), niters);
	erode(temp, temp, Mat(), Point(-1,-1), niters*2);
	dilate(temp, temp, Mat(), Point(-1,-1), niters);
 
	findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
 
	dst = Mat::zeros(img.size(), CV_8UC3);
 
	if( contours.size() == 0 )
		return;
 
	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	int idx = 0, largestComp = 0;
	double maxArea = 0;
 
	for( ; idx >= 0; idx = hierarchy[idx][0] )
	{
		const vector<Point>& c = contours[idx];
		double area = fabs(contourArea(Mat(c)));
		if( area > maxArea )
		{
			maxArea = area;
			largestComp = idx;
		}
	}
	Scalar color( 0, 0, 255 );
	drawContours( dst, contours, largestComp, color, CV_FILLED, 8, hierarchy );
	std::vector<cv::Rect> boundRect(contours.size());
	int max_area = 0;
	int max_index = 0;	
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		if(boundRect[i].area() > max_area) {
			max_area = boundRect[i].area();
			max_index = i;
		}
	}
	if(max_index > 0)
	{
		rectangle(dst, boundRect[max_index], cv::Scalar(0, 255, 0), 2);//在result上绘制正外接矩形
	} 	
}

void RosDetect::cmpRegionDepth(cv::Mat _img)
{
    int ocp_pixels = 0;
    float accm_value = 0.0f;    
    int pixels_num = _img.rows*_img.cols;
    
    std::vector<float> ocp_depths(pixels_num);
    if (!_img.empty()) {
        for (int x = 0; x < _img.cols; x++) {
            for (int y = 0; y < _img.rows; y++) {
                float d = _img.at<float>(y, x);
                if (d > 0.1f && d < 10.0f) {
                    ocp_depths[ocp_pixels] = d;
                    accm_value += d;
                    ocp_pixels++;
                }
            }
        }
    }
    if(ocp_pixels > 0) {
        avg_depth = accm_value / ocp_pixels;
        avg_sat = (float)ocp_pixels / pixels_num;  
#if 1    
        std::cout << "AVG_DEPTH : " << avg_depth << ", avg_sat : " << avg_sat 
                    << ", pixels_num : " << pixels_num << ", ocp_pixels : " << ocp_pixels << std::endl;
#endif
    }
}

int RosDetect::detectService(cv::Rect& rect_)
{
    if(depth_img_.empty())
        depth_img_ = cv::Mat(frame.size(), CV_32FC1);

    cv::Mat roi_depth = depth_img_(rect_);
    cmpRegionDepth(roi_depth);
    if(avg_depth < 8.0 && avg_depth > 4.0 && avg_sat > 0.8) {
        srv.request.xmin = rect_.x;
        srv.request.ymin = rect_.y;
        srv.request.xmax = rect_.x + rect_.width;
        srv.request.ymax = rect_.y + rect_.height;
        if (client_.call(srv))
        {
            ROS_INFO("Sum: %ld", (long int)srv.response.sum);
            return 1;
        }
        else
        {
            ROS_ERROR("Failed to call service add_two_ints");
            return 0;
        }        
    }
    return 0;
}
int RosDetect::start() 
{
    int ret = 0;
    if(frame.empty()) {
        printf("wait ......\n");
        return 0;
    }
    nFrmNum++;
    //如果是第一帧，需要申请内存，并初始化
    if (nFrmNum == 1)
    {
        bkImg = cv::Mat(frame.size(), CV_8UC1);
        frImg = cv::Mat(frame.size(), CV_8UC1);
        frameMat = cv::Mat(frame.size(), CV_32FC1);
        bkMat = cv::Mat(frame.size(), CV_32FC1);
        frMat = cv::Mat(frame.size(), CV_32FC1);
        frcontImg = cv::Mat(frame.size(), CV_8UC1);
        result = cv::Mat(frame.size(), CV_8UC1);
        //convert frame into the grayscale image
        cvtColor(frame, bkImg, CV_BGR2GRAY);
        cvtColor(frame, frImg, CV_BGR2GRAY);
        frImg.convertTo(frameMat, CV_32FC1);
        frImg.convertTo(frMat, CV_32FC1);
        frImg.convertTo(frImg, CV_32FC1);
        kernel_erode = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        kernel_dilate = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), Point(-1, -1));  
        bgsubtractor = createBackgroundSubtractorMOG2();
    }
    else if(nFrmNum > 1)
    {
        result = frame.clone();
        tmp_frame = frame.clone();
        cvtColor(frame, frImg, CV_BGR2GRAY);
        frImg.convertTo(frameMat, CV_32FC1);
        //先做高斯滤波，以平滑图像
        GaussianBlur(frameMat, frameMat, cv::Size(3, 3), 3, 3, 0);

        //当前帧跟背景图相减 
        // cv::absdiff(backgroundImage,currentImage,foreground);
        absdiff(frameMat, bkMat, frMat);
        //二值化前景图
        threshold(frMat, frImg, 20, 255.0, CV_THRESH_BINARY);
        //进行形态学滤波，去掉噪音  
        bgsubtractor->apply(tmp_frame, bgmask);
        morphologyEx(bgmask, bgmask, MORPH_OPEN, kernel_erode, Point(-1, -1));
		refineSegments(tmp_frame, bgmask, out_frame);

        dilate(frImg, frImg, kernel_dilate);
        erode(frImg, frImg, kernel_erode);
        dilate(frImg, frImg, kernel_dilate);
        //update the background
        cv::addWeighted(frameMat, 1 - 0.03, bkMat, 0.03, 0, bkMat);
        bkMat.convertTo(bkImg, CV_8UC1);
        frImg.convertTo(frcontImg, CV_8UC1);

        cv::findContours(frcontImg, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
        cv::drawContours(result, contours, -1, cv::Scalar(0, 0, 255), 2);

        std::vector<cv::Rect> boundRect(contours.size());
        int max_area = 0;
        int max_index = 0;
        // cv::Rect bound_(frImg.cols-1, frImg.rows-1, 1, 1);
        for (int i = 0; i < contours.size(); i++)
        {
            boundRect[i] = boundingRect(contours[i]);
            if(boundRect[i].area() > max_area) {
                max_area = boundRect[i].area();
                max_index = i;
            }
        }
        
        if(max_index > 0)
        {
            // if(rect_depth > 2 && rect_depth < 3 rect_saturation > 0.8)
            {
                ret = detectService(boundRect[max_index]);
            }    
            rectangle(result, boundRect[max_index], cv::Scalar(0, 255, 0), 2);//在result上绘制正外接矩形

        }    
        cv::imshow("video", frame);
        // cv::imshow("background", bkImg);
        cv::imshow("foreground", frImg);
        cv::imshow("result", out_frame);

    } // end of if-else
    return ret;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ros_detect");
    RosDetect detect_;
    ros::Rate loop_rate(30);
    while (ros::ok())
    {
        int ret = detect_.start();
        //如果有按键事件，则跳出循环
        if (waitKey(2) >= 0)
            break;        
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}