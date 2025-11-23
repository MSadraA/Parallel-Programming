#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    cv::Mat in_img, out_img_1, out_img_2;

    in_img = cv::imread("PCB.png", cv::IMREAD_GRAYSCALE);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(0, 0));

    cv::dilate(in_img, out_img_1, element);
    cv::erode(in_img, out_img_2, element);

    cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("dilation", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("erosion", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("output image", cv::WINDOW_AUTOSIZE);

    
    cv::imshow("input image", in_img);
    cv::imshow("dilation", out_img_1);
    cv::imshow("erosion", out_img_2);
    cv::imshow("output image", out_img_1 - out_img_2);

    cv::waitKey(0);
    return 0;
}