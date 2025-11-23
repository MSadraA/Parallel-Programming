#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

int main()
{
    cv::Mat in_img, out_img;
    in_img = cv::imread("Boats.png", cv::IMREAD_GRAYSCALE);

    cv::blur(in_img, out_img, cv::Size(7,7));

    cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("output image", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("input image", in_img);
    cv::imshow("output image", out_img);

    cv::waitKey(0);
    return 0;
}