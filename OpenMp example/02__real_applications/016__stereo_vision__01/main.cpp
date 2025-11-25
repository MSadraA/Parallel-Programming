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

    cv::namedWindow("left image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("right image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("disparity", cv::WINDOW_AUTOSIZE);
    
    imgL = cv::imread("img_left.png",  cv::IMREAD_GRAYSCALE);
    imgR = cv::imread("img_right.png", cv::IMREAD_GRAYSCALE);

    stereo->setNumDisparities(numDisparity);
    stereo->compute(imgL, imgR, disp);
    disp.convertTo(disparity, CV_8U);

    cv::imshow("left image", imgL);
    cv::imshow("right image", imgR);
    cv::imshow("disparity", disparity);

    cv::waitKey(0);
    return 0;
}