#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using std::cout;
using std::endl;

int main()
{
    int numDisparity = 32; 
    int blockSize = 5;
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    cv::Mat disp,disparity; //Disparity
    cv::Mat imgL;
    cv::Mat imgR;
    char imgL_str [256];
    char imgR_str [256];


    cv::namedWindow("left image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("right image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("disparity", cv::WINDOW_AUTOSIZE);
    
    for (int frame_cnt = 0; frame_cnt < 326; frame_cnt++) {
        sprintf (imgL_str, "./Images/Left/left%04d.tif", frame_cnt);
        sprintf (imgR_str, "./Images/right/right%04d.tif", frame_cnt);
        
        imgL = cv::imread(imgL_str,  cv::IMREAD_GRAYSCALE);
        imgR = cv::imread(imgR_str, cv::IMREAD_GRAYSCALE);

        stereo->setNumDisparities(numDisparity);
        stereo->compute(imgL, imgR, disp);
        disp.convertTo(disparity, CV_8U);

        cv::imshow("left image", imgL);
        cv::imshow("right image", imgR);
        cv::imshow("disparity", disparity);

        cv::waitKey(500);
    }

    cv::waitKey(0);
    return 0;
}