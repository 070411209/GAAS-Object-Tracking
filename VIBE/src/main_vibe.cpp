#include <opencv2/opencv.hpp>  
#include "vibe_bgs.h"    
#include <iostream>    
#include <cstdio>    
#include <stdlib.h>  
using namespace cv;    
using namespace std;    
    
int main(int argc, char* argv[])    
{    
    Mat frame, gray, mask;    
    VideoCapture capture;    
    capture.open(0);    //输入视频地址
    
    if (!capture.isOpened())    
    {    
        cout<<"No camera or video input!\n"<<endl;    
        return -1;    
    }    
    
    ViBe_BGS Vibe_Bgs; //定义一个背景差分对象  
    int count = 0; //帧计数器，统计为第几帧   
    
    while (1)    
    {    
        count++;    
        capture >> frame;    
        if (frame.empty())    
            break;    
        cvtColor(frame, gray, CV_RGB2GRAY); //转化为灰度图像   
        
        if (count == 1)  //若为第一帧  
        {    
            Vibe_Bgs.init(gray);    
            Vibe_Bgs.processFirstFrame(gray); //背景模型初始化   
            cout<<" Training GMM complete!"<<endl;    
        }    
        else    
        {    
            Vibe_Bgs.testAndUpdate(gray);    
            mask = Vibe_Bgs.getMask();    //计算前景
            morphologyEx(mask, mask, MORPH_OPEN, Mat());   //形态学处理消除前景图像中的小噪声，这里用的开运算 
            imshow("mask", mask);    
        }    
    
        imshow("input", frame);     
    
        if ( cvWaitKey(10) == 'q' )    //键盘键入q,则停止运行，退出程序
            break;    
    }    
    return 0;    
}