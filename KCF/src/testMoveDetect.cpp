//运动物体检测——帧差法  
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

cv::Mat MoveDetect(cv::Mat temp, cv::Mat frame)
{
    cv::Mat result = frame.clone();
    //1.将background和frame转为灰度图  
    cv::Mat gray1, gray2;
    cv::cvtColor(temp, gray1, CV_BGR2GRAY);
    cv::cvtColor(frame, gray2, CV_BGR2GRAY);
    //2.将background和frame做差
    cv::Mat diff;
    cv::absdiff(gray1, gray2, diff);
    cv::imshow("diff", diff);
    //3.对差值图diff_thresh进行阈值化处理  
    cv::Mat diff_thresh;
    cv::threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
    cv::imshow("diff_thresh", diff_thresh);
    //4.腐蚀
    cv::Mat kernel_erode = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat kernel_dilate = getStructuringElement(cv::MORPH_RECT, cv::Size(18, 18));
    cv::erode(diff_thresh, diff_thresh, kernel_erode);
    cv::imshow("erode", diff_thresh);
    //5.膨胀  
    cv::dilate(diff_thresh, diff_thresh, kernel_dilate);
    cv::imshow("dilate", diff_thresh);
    //6.查找轮廓并绘制轮廓  
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    cv::drawContours(result, contours, -1, cv::Scalar(0, 0, 255), 2);//在result上绘制轮廓  
                                                             //7.查找正外接矩形  
    std::vector<cv::Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        boundRect[i] = boundingRect(contours[i]);
        rectangle(result, boundRect[i], cv::Scalar(0, 255, 0), 2);//在result上绘制正外接矩形  
    }
    return result;//返回result  
}

int main() {
    //打开视频文件：其实就是建立一个VideoCapture结构 
    VideoCapture video(0); 
    //检测是否正常打开:成功打开时，isOpened返回ture  
    if (!video.isOpened())
    {
        return -1;
    }


    int frameCount = 100000;//video.get(CV_CAP_PROP_FRAME_COUNT);//获取帧数
    double FPS = video.get(CV_CAP_PROP_FPS);//获取FPS  
    cv::Mat frame;//存储帧
    cv::Mat temp;//存储前一帧图像  
    cv::Mat result;//存储结果图像  
    for (int i = 0; i < frameCount; i++)
    {
        video >> frame;
        cv::imshow("frame", frame);
        if (frame.empty())//对帧进行异常检测  
        {
            cout << "frame is empty!" << endl;
            break;
        }
        if (i == 0)//如果为第一帧（temp还为空）  
        {
            result = MoveDetect(frame, frame);//调用MoveDetect()进行运动物体检测，返回值存入result  
        }
        else//若不是第一帧（temp有值了）  
        {
            result = MoveDetect(temp, frame);//调用MoveDetect()进行运动物体检测，返回值存入result  

        }
        cv::imshow("result", result);
        if (cv::waitKey(1000.0 / FPS) == 27)//按原FPS显示  
        {
            cout << "ESC退出!" << endl;
            break;
        }
        temp = frame.clone();
    }
    return 0;
}