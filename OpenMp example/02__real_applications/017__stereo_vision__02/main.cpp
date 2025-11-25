#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using std::cout;
using std::endl;

int main()
{
    cv::Mat imgL;
    cv::Mat imgR;
    char imgL_str [256];
    char imgR_str [256];

    cv::namedWindow("left image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("right image", cv::WINDOW_AUTOSIZE);
    
    for (int frame_cnt = 0; frame_cnt < 326; frame_cnt++) {
        sprintf (imgL_str, "./Images/Left/left%04d.tif", frame_cnt);
        sprintf (imgR_str, "./Images/right/right%04d.tif", frame_cnt);
        
        imgL = cv::imread(imgL_str);
        imgR = cv::imread(imgR_str);

        cv::imshow("left image", imgL);
        cv::imshow("right image", imgR);

        cv::waitKey(30);   
    }

    cv::waitKey(0);

    return 0;
}