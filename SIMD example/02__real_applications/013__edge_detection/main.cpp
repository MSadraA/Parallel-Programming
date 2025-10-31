#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    cv::Mat in_img, out_img;
    in_img = cv::imread("Boats.png", cv::IMREAD_GRAYSCALE);

    cv::Canny(in_img, out_img, 50, 200);

    cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("output image", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("input image", in_img);
    cv::imshow("output image", out_img);

    cv::waitKey(0);
    return 0;
}