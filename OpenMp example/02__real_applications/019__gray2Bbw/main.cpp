#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "x86intrin.h"
#include <sys/time.h>

using namespace cv;
using namespace std;

int main( )
{
	cv::Mat in_img = cv::imread("./Boats.png", IMREAD_GRAYSCALE);
	unsigned char *in_image  = (unsigned char *) in_img.data;

	unsigned int NCOLS = in_img.cols;
	unsigned int NROWS = in_img.rows;

	// Convert to BW
	cv::Mat out_img (in_img.rows, in_img.cols, CV_8U);
	unsigned char *out_image = (unsigned char *) out_img.data;

	cv::Mat out_img2 (in_img.rows, in_img.cols, CV_8U);
	unsigned char *out_image2 = (unsigned char *) out_img2.data;

	struct timeval start1, end1;
	gettimeofday(&start1, NULL);
	for (int row = 0; row < NROWS; row++)
		for (int col = 0; col < NCOLS; col++)
			if (*(in_image + row * NCOLS + col) < 127)
				*(out_image + row * NCOLS + col) = 0;
			else
				*(out_image + row * NCOLS + col) = 255;
	gettimeofday(&end1, NULL);
	long seconds_01 = (end1.tv_sec - start1.tv_sec);
	long time_01 = ((seconds_01 * 1000000) + end1.tv_usec) - (start1.tv_usec);


	__m128i *pSrc;
	__m128i *pRes;
	__m128i m1, m2, m3;

	pSrc = (__m128i *) in_img.data;
	pRes = (__m128i *) out_img2.data;

	struct timeval start2, end2;
	gettimeofday(&start2, NULL);
	m2 = _mm_set1_epi8 ((unsigned char) 0XEF);
	for (int i = 0; i < NROWS; i++)
		for (int j = 0; j < NCOLS / 16; j++)
		{
			m1 = _mm_loadu_si128(pSrc + i * NCOLS/16 + j) ;
			m3 = _mm_cmplt_epi8 (m1, m2);
			_mm_storeu_si128 (pRes + i * NCOLS/16 + j, m3);
		}
	gettimeofday(&end2, NULL);
	long seconds_02 = (end2.tv_sec - start2.tv_sec);
	long time_02 = ((seconds_02 * 1000000) + end2.tv_usec) - (start2.tv_usec);

	cv::Mat show_in_img (in_img.rows/2, in_img.cols/2, CV_8U); 
	cv::resize(in_img, show_in_img, cv::Size(), 0.5, 0.5);
	cv::namedWindow("input", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("input", show_in_img);

	cv::Mat show_out_img (out_img.rows/2, out_img.cols/2, CV_8U); 
	cv::resize(out_img, show_out_img, cv::Size(), 0.5, 0.5);
	cv::namedWindow("output(serial)", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("output(serial)", show_out_img);

	cv::Mat show_out_img2 (out_img2.rows/2, out_img2.cols/2, CV_8U); 
	cv::resize(out_img2, show_out_img2, cv::Size(), 0.5, 0.5);
	cv::namedWindow("output(paralal)", cv::WINDOW_AUTOSIZE); 	
	cv::imshow("output(paralal)", show_out_img2);

	waitKey(0);

	printf ("Serial Run time = %ld \n", time_01);
	printf ("Parallel Run time = %ld \n", time_02);
	printf ("Speedup = %4.2f\n", (float) (time_01)/(float) time_02);

	return 0;
}
