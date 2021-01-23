// background_segment_objects.cpp : 定义控制台应用程序的入口点。
//
 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <string>
 
using namespace cv;
using namespace std;

static void help()
{
	printf("\n"
		"This program demonstrated a simple method of connected components clean up of background subtraction\n"
		"When the program starts, it begins learning the background.\n"
		"You can toggle background learning on and off by hitting the space bar.\n"
		"Call\n"
		"./segment_objects [video file, else it reads camera 0]\n\n");
}
 
static void refineSegments(const Mat& img, Mat& mask, Mat& dst)
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
 
 
int main(int argc, char** argv)
{
	VideoCapture cap(0);
	bool update_bg_model = true;
 
	Mat tmp_frame, bgmask, out_frame;
 
	cap >> tmp_frame;
	if(!tmp_frame.data)
	{
		printf("can not read data from the video source\n");
		return -1;
	}
 
	namedWindow("video", 1);
	namedWindow("segmented", 1);
 
	Ptr<BackgroundSubtractor> bgsubtractor = createBackgroundSubtractorMOG2();
 	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	for(;;)
	{
		cap >> tmp_frame;
		if( !tmp_frame.data )
			break;

        bgsubtractor->apply(tmp_frame, bgmask);
        morphologyEx(bgmask, bgmask, MORPH_OPEN, kernel, Point(-1, -1));
		refineSegments(tmp_frame, bgmask, out_frame);
		imshow("video", tmp_frame);
		imshow("segmented", out_frame);
		int keycode = waitKey(30);
		if( keycode == 27 )
			break;
		if( keycode == ' ' )
		{
			update_bg_model = !update_bg_model;
			printf("Learn background is in state = %d\n",update_bg_model);
		}
	}
 
	return 0;
}
 