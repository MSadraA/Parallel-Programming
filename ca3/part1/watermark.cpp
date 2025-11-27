#include <iostream>
#include <iomanip>
#include <vector>
#include "ipp.h"       
#include <immintrin.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void simd_calculation(const Mat& base_img, const Mat& water_img, Mat& out_img) {
    const int height = base_img.rows;
    const int width = base_img.cols;
    const float total_dim_inv = 1.0f / (float)(width + height);

    // Start parallel region
    __m256 constant_vec = _mm256_set1_ps(total_dim_inv);
    __m256 one_vec = _mm256_set1_ps(1.0f);
    __m256 x_index = _mm256_set_ps(7.0 , 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0); // 0.0 to smallest index (i)
    __m256 x_step = _mm256_set1_ps(8.0f * total_dim_inv);

    for (int y = 0; y < height; ++y) {
        // access to current pixel
        const unsigned char* base_ptr = base_img.ptr<unsigned char>(y);
        unsigned char* out_ptr = out_img.ptr<unsigned char>(y);
        const unsigned char* water_ptr = water_img.ptr<unsigned char>(y);

        __m256 alpha_by_y = _mm256_set1_ps((float)y * total_dim_inv); // y/(height + width)
        __m256 x_index_mul_by_constant = _mm256_mul_ps(x_index, constant_vec); // i/(height + width)
        __m256 alpha_by_x_y = _mm256_add_ps(alpha_by_y, x_index_mul_by_constant); // i/(height + width) + y/(height + width)

        int x = 0;
        const int limit = (width / 8) * 8;

        for(; x < limit; x += 8){
            __m256 one_minus_alpha_vec = _mm256_sub_ps(one_vec, alpha_by_x_y); // 1 - alpha
            __m128i v_base_u8  = _mm_loadl_epi64((__m128i const*)(base_ptr + x));
            __m128i v_water_u8 = _mm_loadl_epi64((__m128i const*)(water_ptr + x));

            __m256i v_base_i32  = _mm256_cvtepu8_epi32(v_base_u8); // it can be done with a single instruction
            __m256i v_water_i32 = _mm256_cvtepu8_epi32(v_water_u8);
            __m256 v_base_f  = _mm256_cvtepi32_ps(v_base_i32);
            __m256 v_water_f = _mm256_cvtepi32_ps(v_water_i32);
            
            // calculate result for 8 pixels
            __m256 term1 = _mm256_mul_ps(v_water_f, alpha_by_x_y);
            __m256 term2 = _mm256_mul_ps(v_base_f, one_minus_alpha_vec);
            __m256 v_result_f = _mm256_add_ps(term1, term2);

            // convert result to unsigned char
            __m256i v_result_i32 = _mm256_cvtps_epi32(v_result_f); // convert to 32-bit integer
            __m128i v_low_i32  = _mm256_castsi256_si128(v_result_i32); // extract lower 128 bits
            __m128i v_high_i32 = _mm256_extractf128_si256(v_result_i32, 1); // extract upper 128 bits
            __m128i v_result_i16 = _mm_packus_epi32(v_low_i32, v_high_i32); // pack unsigned 16-bit integers
            __m128i v_result_u8  = _mm_packus_epi16(v_result_i16, _mm_setzero_si128()); // pack unsigned 8-bit integers
            _mm_storel_epi64((__m128i*)(out_ptr + x), v_result_u8); // store packed unsigned 8-bit integers

            // update alpha for next 8 pixels
            alpha_by_x_y = _mm256_add_ps(alpha_by_x_y, x_step); // new alpha = i/(height + width) + y/(height + width) + 8/(height + width)
        }

        // calculate remaining pixels
        for (; x < width; ++x) {
            float alpha = ((float)x + (float)y) * total_dim_inv;
            float one_minus_alpha = 1.0f - alpha;
            float base_f = (float)base_ptr[x];
            float water_f = (float)water_ptr[x];
            float out_f = (water_f * alpha) + (base_f * one_minus_alpha);
            out_ptr[x] = (unsigned char)out_f;
        }
    }
}

void serial_optimized_calculation(const Mat& base_img, const Mat& water_img, Mat& out_img) 
{
    const int height = base_img.rows;
    const int width = base_img.cols;
    const float total_dim_inv = 1.0f / (float)(width + height);

    const float c = total_dim_inv;
    const float x_indices[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    const float x_step = 8.0f * c;
    const float one = 1.0f;

    float a0, a1, a2, a3, a4, a5, a6, a7;
    float out_f0, out_f1, out_f2, out_f3, out_f4, out_f5, out_f6, out_f7;

    for (int y = 0; y < height; y++) {
        const unsigned char* base_ptr = base_img.ptr<unsigned char>(y);
        const unsigned char* water_ptr = water_img.ptr<unsigned char>(y);
        unsigned char* out_ptr = out_img.ptr<unsigned char>(y);
        
        const float alpha_y = (float)y * c;
        a0 = alpha_y + (x_indices[0] * c);
        a1 = alpha_y + (x_indices[1] * c);
        a2 = alpha_y + (x_indices[2] * c);
        a3 = alpha_y + (x_indices[3] * c);
        a4 = alpha_y + (x_indices[4] * c);
        a5 = alpha_y + (x_indices[5] * c);
        a6 = alpha_y + (x_indices[6] * c);
        a7 = alpha_y + (x_indices[7] * c);

        int x = 0;
        const int limit = (width / 8) * 8;

        for (; x < limit; x += 8) {
            
            out_f0 = (float)water_ptr[x+0] * a0 + (float)base_ptr[x+0] * (one - a0);
            out_f1 = (float)water_ptr[x+1] * a1 + (float)base_ptr[x+1] * (one - a1);
            out_f2 = (float)water_ptr[x+2] * a2 + (float)base_ptr[x+2] * (one - a2);
            out_f3 = (float)water_ptr[x+3] * a3 + (float)base_ptr[x+3] * (one - a3);
            out_f4 = (float)water_ptr[x+4] * a4 + (float)base_ptr[x+4] * (one - a4);
            out_f5 = (float)water_ptr[x+5] * a5 + (float)base_ptr[x+5] * (one - a5);
            out_f6 = (float)water_ptr[x+6] * a6 + (float)base_ptr[x+6] * (one - a6);
            out_f7 = (float)water_ptr[x+7] * a7 + (float)base_ptr[x+7] * (one - a7);
            
            out_ptr[x+0] = (unsigned char)out_f0;
            out_ptr[x+1] = (unsigned char)out_f1;
            out_ptr[x+2] = (unsigned char)out_f2;
            out_ptr[x+3] = (unsigned char)out_f3;
            out_ptr[x+4] = (unsigned char)out_f4;
            out_ptr[x+5] = (unsigned char)out_f5;
            out_ptr[x+6] = (unsigned char)out_f6;
            out_ptr[x+7] = (unsigned char)out_f7;

            a0 += x_step;
            a1 += x_step;
            a2 += x_step;
            a3 += x_step;
            a4 += x_step;
            a5 += x_step;
            a6 += x_step;
            a7 += x_step;
        }

        for (; x < width; ++x) {
            float alpha = ((float)x + (float)y) * total_dim_inv;
            float one_minus_alpha = 1.0f - alpha;
            float base_f = (float)base_ptr[x];
            float water_f = (float)water_ptr[x];
            float out_f = (water_f * alpha) + (base_f * one_minus_alpha); 
            out_ptr[x] = (unsigned char)out_f;
        }
    }
}

void openmp_calculation(const Mat& base_img, const Mat& water_img, Mat& out_img) 
{
    const int height = base_img.rows;
    const int width = base_img.cols;
    const float total_dim_inv = 1.0f / (float)(width + height);

    const float c = total_dim_inv;
    const float x_indices[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    const float x_step = 8.0f * c;
    const float one = 1.0f;

    #pragma omp parallel for schedule(static) proc_bind(close)
    for (int y = 0; y < height; y++) {
        
        const unsigned char* base_ptr = base_img.ptr<unsigned char>(y);
        const unsigned char* water_ptr = water_img.ptr<unsigned char>(y);
        unsigned char* out_ptr = out_img.ptr<unsigned char>(y);
        
        float a0, a1, a2, a3, a4, a5, a6, a7;
        float out_f0, out_f1, out_f2, out_f3, out_f4, out_f5, out_f6, out_f7;

        const float alpha_y = (float)y * c;
        
        a0 = alpha_y + (x_indices[0] * c);
        a1 = alpha_y + (x_indices[1] * c);
        a2 = alpha_y + (x_indices[2] * c);
        a3 = alpha_y + (x_indices[3] * c);
        a4 = alpha_y + (x_indices[4] * c);
        a5 = alpha_y + (x_indices[5] * c);
        a6 = alpha_y + (x_indices[6] * c);
        a7 = alpha_y + (x_indices[7] * c);

        int x = 0;
        const int limit = (width / 8) * 8;

        for (; x < limit; x += 8) {
            
            out_f0 = (float)water_ptr[x+0] * a0 + (float)base_ptr[x+0] * (one - a0);
            out_f1 = (float)water_ptr[x+1] * a1 + (float)base_ptr[x+1] * (one - a1);
            out_f2 = (float)water_ptr[x+2] * a2 + (float)base_ptr[x+2] * (one - a2);
            out_f3 = (float)water_ptr[x+3] * a3 + (float)base_ptr[x+3] * (one - a3);
            out_f4 = (float)water_ptr[x+4] * a4 + (float)base_ptr[x+4] * (one - a4);
            out_f5 = (float)water_ptr[x+5] * a5 + (float)base_ptr[x+5] * (one - a5);
            out_f6 = (float)water_ptr[x+6] * a6 + (float)base_ptr[x+6] * (one - a6);
            out_f7 = (float)water_ptr[x+7] * a7 + (float)base_ptr[x+7] * (one - a7);
            
            out_ptr[x+0] = (unsigned char)out_f0;
            out_ptr[x+1] = (unsigned char)out_f1;
            out_ptr[x+2] = (unsigned char)out_f2;
            out_ptr[x+3] = (unsigned char)out_f3;
            out_ptr[x+4] = (unsigned char)out_f4;
            out_ptr[x+5] = (unsigned char)out_f5;
            out_ptr[x+6] = (unsigned char)out_f6;
            out_ptr[x+7] = (unsigned char)out_f7;

            a0 += x_step; a1 += x_step; a2 += x_step; a3 += x_step;
            a4 += x_step; a5 += x_step; a6 += x_step; a7 += x_step;
        }

        for (; x < width; ++x) {
            float alpha = ((float)x + (float)y) * total_dim_inv;
            float out_f = ((float)water_ptr[x] * alpha) + ((float)base_ptr[x] * (1.0f - alpha)); 
            out_ptr[x] = (unsigned char)out_f;
        }
    }
}

void openmp_simd_calculation(const Mat& base_img, const Mat& water_img, Mat& out_img) 
{
    const int height = base_img.rows;
    const int width = base_img.cols;
    const float total_dim_inv = 1.0f / (float)(width + height);

    const __m256 constant_vec = _mm256_set1_ps(total_dim_inv);
    const __m256 one_vec = _mm256_set1_ps(1.0f);
    const __m256 x_index_base = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f); 
    const __m256 x_step = _mm256_set1_ps(8.0f * total_dim_inv);

    #pragma omp parallel for schedule(static) proc_bind(close)
    for (int y = 0; y < height; ++y) {
        
        const unsigned char* base_ptr = base_img.ptr<unsigned char>(y);
        const unsigned char* water_ptr = water_img.ptr<unsigned char>(y);
        unsigned char* out_ptr = out_img.ptr<unsigned char>(y);

        __m256 alpha_by_y = _mm256_set1_ps((float)y * total_dim_inv);
        __m256 x_index_mul_by_constant = _mm256_mul_ps(x_index_base, constant_vec);
        __m256 alpha_by_x_y = _mm256_add_ps(alpha_by_y, x_index_mul_by_constant);

        int x = 0;
        const int limit = (width / 8) * 8;

        for(; x < limit; x += 8){
            __m256 one_minus_alpha_vec = _mm256_sub_ps(one_vec, alpha_by_x_y);

            __m128i v_base_u8  = _mm_loadl_epi64((__m128i const*)(base_ptr + x));
            __m128i v_water_u8 = _mm_loadl_epi64((__m128i const*)(water_ptr + x));

            __m256i v_base_i32  = _mm256_cvtepu8_epi32(v_base_u8);
            __m256i v_water_i32 = _mm256_cvtepu8_epi32(v_water_u8);
            __m256 v_base_f  = _mm256_cvtepi32_ps(v_base_i32);
            __m256 v_water_f = _mm256_cvtepi32_ps(v_water_i32);
            
            // Calculate: (water * alpha) + (base * (1-alpha))
            __m256 term1 = _mm256_mul_ps(v_water_f, alpha_by_x_y);
            __m256 term2 = _mm256_mul_ps(v_base_f, one_minus_alpha_vec);
            __m256 v_result_f = _mm256_add_ps(term1, term2);

            // Compress & Store
            __m256i v_result_i32 = _mm256_cvtps_epi32(v_result_f);
            __m128i v_low_i32  = _mm256_castsi256_si128(v_result_i32);
            __m128i v_high_i32 = _mm256_extractf128_si256(v_result_i32, 1);
            __m128i v_result_i16 = _mm_packus_epi32(v_low_i32, v_high_i32);
            __m128i v_result_u8  = _mm_packus_epi16(v_result_i16, _mm_setzero_si128());
            
            _mm_storel_epi64((__m128i*)(out_ptr + x), v_result_u8);

            // Update Alpha
            alpha_by_x_y = _mm256_add_ps(alpha_by_x_y, x_step);
        }

        for (; x < width; ++x) {
            float alpha = ((float)x + (float)y) * total_dim_inv;
            float out_f = ((float)water_ptr[x] * alpha) + ((float)base_ptr[x] * (1.0f - alpha));
            out_ptr[x] = (unsigned char)out_f;
        }
    }
}

int main() {
    cv::setNumThreads(0);
    Ipp64u start, end;
    Ipp64u time1 , time2 , time3 , time4;

    Mat base_img = imread("./assets/base.jpg", IMREAD_COLOR);
    Mat water_img = imread("./assets/watermark.png", IMREAD_COLOR);

    if (base_img.empty() || water_img.empty()) {
        cout << "Error: Could not load images." << endl;
        return 1;
    }

    resize(water_img, water_img, base_img.size());

    vector<Mat> base_channels(3), water_channels(3);

    vector<Mat> out_parallel_channels(3);
    vector<Mat> out_serial_channels(3);
    vector<Mat> out_openmp_channels(3);
    vector<Mat> out_openmp_simd_channels(3);

    split(base_img, base_channels);
    split(water_img, water_channels);
    for(int i=0; i<3; ++i) {
        out_parallel_channels[i] = Mat::zeros(base_img.size(), CV_8UC1);
        out_serial_channels[i] = Mat::zeros(base_img.size(), CV_8UC1);
        out_openmp_channels[i] = Mat::zeros(base_img.size(), CV_8UC1);
        out_openmp_simd_channels[i] = Mat::zeros(base_img.size(), CV_8UC1);
    }
    const float total_dim = 1.0f / (float)(base_img.cols + base_img.rows);

    // -- serial execution ---
    start = ippGetCpuClocks();
    for(int i=0; i<3; ++i) {
        serial_optimized_calculation(base_channels[i], water_channels[i], out_serial_channels[i]);
    }
    end = ippGetCpuClocks();
    time2 = end - start;

    // -- parallel execution ---
    start = ippGetCpuClocks();
    
    for(int i=0; i<3; ++i) {
        simd_calculation(base_channels[i], water_channels[i], out_parallel_channels[i]);
    }
    
    end = ippGetCpuClocks();
    time1 = end - start;

    // -- openmp execution ---
    start = ippGetCpuClocks();
    for(int i=0; i<3; ++i) {
        openmp_calculation(base_channels[i], water_channels[i], out_openmp_channels[i]);
    }
    end = ippGetCpuClocks();
    time3 = end - start;

    // -- openmp + simd execution ---
    start = ippGetCpuClocks();
    for(int i=0; i<3; ++i) {
        openmp_simd_calculation(base_channels[i], water_channels[i], out_openmp_simd_channels[i]);
    }
    end = ippGetCpuClocks();
    time4 = end - start;


    // --- final result ---
    Mat out_serial, out_parallel , out_openmp , out_openmp_simd;
    merge(out_serial_channels, out_serial);
    merge(out_parallel_channels, out_parallel);
    merge(out_openmp_channels, out_openmp);
    merge(out_openmp_simd_channels, out_openmp_simd);
    
    cout << "--- Watermark Result ---" << endl;
    cout << std::fixed << std::setprecision(4);
    cout << "Serial (Optimized) clock cycles: " << (long long)time2 << endl;
    cout << "Parallel (AVX) clock cycles:     " << (long long)time1 << endl;
    cout << "OpenMP clock cycles:             " << (long long)time3 << endl;
    cout << "OpenMP + SIMD clock cycles:      " << (long long)time4 << endl;
    cout << "SIMD speedup: " << ((double)time2 / (double)time1) << "x" << endl;
    cout << "OpenMP speedup: " << ((double)time2 / (double)time3) << "x" << endl;
    cout << "OpenMP + SIMD speedup: " << ((double)time2 / (double)time4) << "x" << endl;

    imwrite("watermark_output_serial.jpg", out_serial);
    imwrite("watermark_output_parallel.jpg", out_parallel);
    imwrite("watermark_output_openmp.jpg", out_openmp);
    imwrite("watermark_output_openmp_simd.jpg", out_openmp_simd);
    cout << "\nOutput images saved." << endl;
    return 0;
}