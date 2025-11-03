#include <iostream>
#include <iomanip>
#include <vector>
#include "ipp.h"
#include <immintrin.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

static inline void prefetchL1(const void* ptr) {
    _mm_prefetch((const char*)ptr, _MM_HINT_T0);
}

inline void serial_blend_manual(
    const Mat& __restrict base_img,
    const Mat& __restrict water_img,
    Mat& __restrict out_img)
{
    const int height = base_img.rows;
    const int width  = base_img.cols;
    const float total_dim_inv = 1.0f / (float)(width + height);
    const float x_step = 8.0f * total_dim_inv;
    const float one = 1.0f;

    // ثابت‌ها برای جلوگیری از reallocation هر بار
    const float idx[8] = {0,1,2,3,4,5,6,7};

    for (int y = 0; y < height; ++y) {
        const unsigned char* __restrict base_ptr  = base_img.ptr<unsigned char>(y);
        const unsigned char* __restrict water_ptr = water_img.ptr<unsigned char>(y);
        unsigned char* __restrict out_ptr         = out_img.ptr<unsigned char>(y);

        const float alpha_y = (float)y * total_dim_inv;

        // precompute 8 α مقادیر اول
        float a0 = alpha_y + idx[0]*total_dim_inv;
        float a1 = alpha_y + idx[1]*total_dim_inv;
        float a2 = alpha_y + idx[2]*total_dim_inv;
        float a3 = alpha_y + idx[3]*total_dim_inv;
        float a4 = alpha_y + idx[4]*total_dim_inv;
        float a5 = alpha_y + idx[5]*total_dim_inv;
        float a6 = alpha_y + idx[6]*total_dim_inv;
        float a7 = alpha_y + idx[7]*total_dim_inv;

        const int limit = (width / 8) * 8;

        // Unrolled inner loop
        for (int x = 0; x < limit; x += 8) {
            // محاسبه‌ی ۸ پیکسل در یک iteration
            const float base0 = (float)base_ptr[x+0];
            const float base1 = (float)base_ptr[x+1];
            const float base2 = (float)base_ptr[x+2];
            const float base3 = (float)base_ptr[x+3];
            const float base4 = (float)base_ptr[x+4];
            const float base5 = (float)base_ptr[x+5];
            const float base6 = (float)base_ptr[x+6];
            const float base7 = (float)base_ptr[x+7];

            const float water0 = (float)water_ptr[x+0];
            const float water1 = (float)water_ptr[x+1];
            const float water2 = (float)water_ptr[x+2];
            const float water3 = (float)water_ptr[x+3];
            const float water4 = (float)water_ptr[x+4];
            const float water5 = (float)water_ptr[x+5];
            const float water6 = (float)water_ptr[x+6];
            const float water7 = (float)water_ptr[x+7];

            // استفاده از FMA برای دقت و سرعت بیشتر
            out_ptr[x+0] = (unsigned char)fmaf(water0, a0, base0 * (one - a0));
            out_ptr[x+1] = (unsigned char)fmaf(water1, a1, base1 * (one - a1));
            out_ptr[x+2] = (unsigned char)fmaf(water2, a2, base2 * (one - a2));
            out_ptr[x+3] = (unsigned char)fmaf(water3, a3, base3 * (one - a3));
            out_ptr[x+4] = (unsigned char)fmaf(water4, a4, base4 * (one - a4));
            out_ptr[x+5] = (unsigned char)fmaf(water5, a5, base5 * (one - a5));
            out_ptr[x+6] = (unsigned char)fmaf(water6, a6, base6 * (one - a6));
            out_ptr[x+7] = (unsigned char)fmaf(water7, a7, base7 * (one - a7));

            // افزایش دستی α
            a0 += x_step; a1 += x_step; a2 += x_step; a3 += x_step;
            a4 += x_step; a5 += x_step; a6 += x_step; a7 += x_step;
        }

        // باقی‌مانده پیکسل‌ها
        for (int x = limit; x < width; ++x) {
            const float alpha = ((float)x + (float)y) * total_dim_inv;
            out_ptr[x] = (unsigned char)fmaf(water_ptr[x], alpha, base_ptr[x] * (1.0f - alpha));
        }
    }
}


__attribute__((always_inline))
inline void blend_channel_avx_manual(
    const Mat& __restrict base_img,
    const Mat& __restrict water_img,
    Mat& __restrict out_img)
{
    const int height = base_img.rows;
    const int width = base_img.cols;
    const size_t step = base_img.step[0];

    const float total_dim_inv = 1.0f / (float)(width + height);

    const __m256 inv_vec  = _mm256_set1_ps(total_dim_inv);
    const __m256 one_vec  = _mm256_set1_ps(1.0f);
    const __m256 step_vec = _mm256_set1_ps(8.0f * total_dim_inv);
    const __m256 base_x   = _mm256_mul_ps(inv_vec, _mm256_set_ps(7,6,5,4,3,2,1,0));

    const int limit = (width / 8) * 8;

    for (int y = 0; y < height; ++y) {
        const unsigned char* __restrict base_ptr  = base_img.data + y * step;
        const unsigned char* __restrict water_ptr = water_img.data + y * step;
        unsigned char* __restrict out_ptr         = out_img.data + y * step;

        __m256 alpha_by_y = _mm256_set1_ps((float)y * total_dim_inv);
        __m256 alpha_vec  = _mm256_add_ps(alpha_by_y, base_x);

        for (int x = 0; x < limit; x += 8) {
            // prefetch future data
            prefetchL1(base_ptr + x + 64);
            prefetchL1(water_ptr + x + 64);

            const __m128i v_base_u8  = _mm_loadl_epi64((__m128i const*)(base_ptr + x));
            const __m128i v_water_u8 = _mm_loadl_epi64((__m128i const*)(water_ptr + x));

            const __m256 v_base_f  = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v_base_u8));
            const __m256 v_water_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v_water_u8));

            const __m256 one_minus_a = _mm256_sub_ps(one_vec, alpha_vec);

            // FMA: (water * α) + (base * (1-α))
            const __m256 v_result = _mm256_fmadd_ps(v_water_f, alpha_vec,
                                                    _mm256_mul_ps(v_base_f, one_minus_a));

            // convert back to 8-bit unsigned
            const __m256i v_i32 = _mm256_cvtps_epi32(v_result);
            const __m128i v_lo = _mm256_castsi256_si128(v_i32);
            const __m128i v_hi = _mm256_extractf128_si256(v_i32, 1);
            const __m128i v_i16 = _mm_packus_epi32(v_lo, v_hi);
            const __m128i v_u8  = _mm_packus_epi16(v_i16, _mm_setzero_si128());
            _mm_storel_epi64((__m128i*)(out_ptr + x), v_u8);

            alpha_vec = _mm256_add_ps(alpha_vec, step_vec);
        }

        // handle remaining pixels
        for (int x = limit; x < width; ++x) {
            float alpha = ((float)x + (float)y) * total_dim_inv;
            out_ptr[x] = (unsigned char)fmaf(alpha, water_ptr[x], base_ptr[x] * (1.0f - alpha));
        }
    }
}

int main() {
    Ipp64u start, end;
    Ipp64u time_avx, time_serial;

    // Align allocations to 32 bytes for AVX
    Mat base_img = imread("./assets/base.jpg", IMREAD_COLOR);
    Mat water_img = imread("./assets/watermark.png", IMREAD_COLOR);

    if (base_img.empty() || water_img.empty()) {
        cout << "Error: Could not load images.\n";
        return 1;
    }

    resize(water_img, water_img, base_img.size());

    vector<Mat> base_channels(3), water_channels(3), out_channels(3);
    split(base_img, base_channels);
    split(water_img, water_channels);
    for (int i=0; i<3; ++i)
        out_channels[i] = Mat::zeros(base_img.size(), CV_8UC1);

    __asm__ __volatile__ ("cpuid" ::: "%rax", "%rbx", "%rcx", "%rdx");
    start = ippGetCpuClocks();

    // process 3 channels with full optimization manually
    for (int i=0; i<3; ++i)
        blend_channel_avx_manual(base_channels[i], water_channels[i], out_channels[i]);

    end = ippGetCpuClocks();
    __asm__ __volatile__ ("cpuid" ::: "%rax", "%rbx", "%rcx", "%rdx");
    time_avx = end - start;


    start = ippGetCpuClocks();
    // process 3 channels with full optimization
    for (int i=0; i<3; ++i)
        serial_blend_manual(base_channels[i], water_channels[i], out_channels[i]);

    end = ippGetCpuClocks();
    time_serial = end - start;


    Mat out_img;
    merge(out_channels, out_img);
    imwrite("watermark_output_manual.jpg", out_img);

    cout << fixed << setprecision(4);
    cout << "--- Manual Optimized Watermark (No Compiler Opts) ---\n";
    cout << "Clock cycles: " << (long long)time_avx << "\n";
    cout << "Clock cycles (serial): " << (long long)time_serial << "\n";
    cout << "Speedup: " << (float)time_serial / (float)time_avx << "\n";
    return 0;
}
