#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        printf("\nCan not open camera \n");
        return -1;
    }
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));

    // intialization BS
#if 1    
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();
#else    
    Ptr<BackgroundSubtractor> pKNN = createBackgroundSubtractorKNN();
#endif
    Mat tmp_frame;
    Mat bsmaskMOG2, bsmaskKNN;

    for (;;)
    {
        cap >> tmp_frame;
        if (tmp_frame.empty())
            break;

#if 1             
        // MOG BS
        pMOG2->apply(tmp_frame, bsmaskMOG2);
        morphologyEx(bsmaskMOG2, bsmaskMOG2, MORPH_OPEN, kernel, Point(-1, -1));
        imshow("MOG2", bsmaskMOG2);
#else
        // KNN BS mask
        pKNN->apply(tmp_frame, bsmaskKNN);
        imshow("KNN Model", bsmaskKNN);
#endif
        imshow("video", tmp_frame);

        char keycode = (char)waitKey(30); //按ESC推出
        if (keycode == 27)
            break;
    }
    cap.release();
    return 0;
}