#include <iostream>
#include <string.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "BackgroundSubtract.h"

using namespace std;

int main(int argc, char* argv[])
{
	cv::VideoCapture cap(0);
	if(cap.isOpened()==NULL)
	{
		cout<<"shit"<<endl;
		return -1;
	}

	cv::Mat image, imageGray;
	cv::Mat foregroundGray;
	BackgroundSubtract subtractor;

	cap>>image;
	cv::cvtColor(image, imageGray, CV_BGR2GRAY);
	imageGray.copyTo(foregroundGray);
	subtractor.init(imageGray);

	while(!image.empty())
	{
		subtractor.subtract(imageGray, foregroundGray);

		imshow("foreground", foregroundGray);
		cap>>image;
		cv::cvtColor(image, imageGray, CV_BGR2GRAY);
		
		int c = cv::waitKey(20);
		if(c==27)
		{
			break;
		}
	}

	return 0;
}