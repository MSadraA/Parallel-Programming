#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    cv::VideoCapture cap = cv::VideoCapture(0);
    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);

    for (;;) {
        cv::Mat frame, gray_frame, edge_frame;
        cap.read(frame);
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::Canny(gray_frame, edge_frame, 100, 200);

        cv::imshow("Edge", edge_frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    return 0;
}