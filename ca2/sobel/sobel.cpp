#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <x86intrin.h> // For __rdtsc()
#include <opencv2/opencv.hpp> // OpenCV Library
#include <immintrin.h> // Main header for AVX

using namespace std;
using namespace cv;

// --- Filter Kernels (float) ---
// Gaussian Kernel
const float gaussian_kernel[9] = {
    1.0f, 2.0f, 1.0f,
    2.0f, 4.0f, 2.0f,
    1.0f, 2.0f, 1.0f
};
const float GAUSSIAN_NORM = 16.0f;

// Sobel X Kernel
const float sobel_x_kernel[9] = {
    -1.0f, 0.0f, 1.0f,
    -2.0f, 0.0f, 2.0f,
    -1.0f, 0.0f, 1.0f
};

// Sobel Y Kernel
const float sobel_y_kernel[9] = {
    -1.0f, -2.0f, -1.0f,
     0.0f,  0.0f,  0.0f,
     1.0f,  2.0f,  1.0f
};

/**
 * @brief Serial 3x3 convolution (Used for Gaussian)
 */
void convolve_serial(const Mat& src, Mat& dst, const float* kernel, float norm_factor) {
    int rows = src.rows;
    int cols = src.cols;
    dst.create(rows, cols, CV_32F);
    
    // Set borders to zero (or handle them)
    dst.setTo(Scalar(0.0));

    for (int y = 1; y < rows - 1; ++y) {
        // Pointers to input rows
        const float* p_top = src.ptr<float>(y - 1); // Top row
        const float* p_mid = src.ptr<float>(y);     // Middle row
        const float* p_bot = src.ptr<float>(y + 1); // Bottom row
        
        // Pointer to output row
        float* p_dst = dst.ptr<float>(y);

        for (int x = 1; x < cols - 1; ++x) {
            
            // Apply 3x3 convolution
            float sum = 0.0f;
            
            // Top row of kernel
            sum += p_top[x - 1] * kernel[0];
            sum += p_top[x]     * kernel[1];
            sum += p_top[x + 1] * kernel[2];
            
            // Middle row of kernel
            sum += p_mid[x - 1] * kernel[3];
            sum += p_mid[x]     * kernel[4];
            sum += p_mid[x + 1] * kernel[5];
            
            // Bottom row of kernel
            sum += p_bot[x - 1] * kernel[6];
            sum += p_bot[x]     * kernel[7];
            sum += p_bot[x + 1] * kernel[8];
            
            // Normalize and store in output
            p_dst[x] = sum / norm_factor;
        }
    }
}

/**
 * @brief Parallel 3x3 convolution (Used for Gaussian)
 */
void convolve_parallel(const Mat& src, Mat& dst, const float* kernel, float norm_factor) {
    int rows = src.rows;
    int cols = src.cols;
    dst.create(rows, cols, CV_32F);
    
    // Set borders to zero
    dst.setTo(Scalar(0.0));

    // --- 1. Prepare Kernel Registers ---
    // Create 9 AVX registers, each holding one kernel weight 8 times
    const __m256 k_vec_0 = _mm256_set1_ps(kernel[0]);
    const __m256 k_vec_1 = _mm256_set1_ps(kernel[1]);
    const __m256 k_vec_2 = _mm256_set1_ps(kernel[2]);
    const __m256 k_vec_3 = _mm256_set1_ps(kernel[3]);
    const __m256 k_vec_4 = _mm256_set1_ps(kernel[4]);
    const __m256 k_vec_5 = _mm256_set1_ps(kernel[5]);
    const __m256 k_vec_6 = _mm256_set1_ps(kernel[6]);
    const __m256 k_vec_7 = _mm256_set1_ps(kernel[7]);
    const __m256 k_vec_8 = _mm256_set1_ps(kernel[8]);
    
    const __m256 k_vec_norm = _mm256_set1_ps(norm_factor);

    // Calculate the limit for the parallel loop
    // We process 8 floats at a time. We stop 8+1 floats from the end.
    const int limit = cols - 9; // (cols - 1) - 8

    for (int y = 1; y < rows - 1; ++y) {
        const float* p_top = src.ptr<float>(y - 1);
        const float* p_mid = src.ptr<float>(y);
        const float* p_bot = src.ptr<float>(y + 1);
        float* p_dst = dst.ptr<float>(y);
        
        int x = 1;
        
        // --- 2. Parallel Loop (8 pixels at a time) ---
        for (; x <= limit; x += 8) {
            // Load 8 pixels from each neighbor position
            const __m256 p0 = _mm256_loadu_ps(&p_top[x - 1]);
            const __m256 p1 = _mm256_loadu_ps(&p_top[x]);
            const __m256 p2 = _mm256_loadu_ps(&p_top[x + 1]);
            const __m256 p3 = _mm256_loadu_ps(&p_mid[x - 1]);
            const __m256 p4 = _mm256_loadu_ps(&p_mid[x]);
            const __m256 p5 = _mm256_loadu_ps(&p_mid[x + 1]);
            const __m256 p6 = _mm256_loadu_ps(&p_bot[x - 1]);
            const __m256 p7 = _mm256_loadu_ps(&p_bot[x]);
            const __m256 p8 = _mm256_loadu_ps(&p_bot[x + 1]);

            // Initialize sum register for 8 pixels to zero
            __m256 sum_vec = _mm256_setzero_ps();

            // --- 3. Multiply and Add (Old Method, as requested) ---
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(p0, k_vec_0));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(p1, k_vec_1));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(p2, k_vec_2));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(p3, k_vec_3));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(p4, k_vec_4));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(p5, k_vec_5));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(p6, k_vec_6));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(p7, k_vec_7));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(p8, k_vec_8));
            
            // --- Fused Multiply-Add ---
            // sum_vec = _mm256_fmadd_ps(p0, k_vec_0, sum_vec);
            // sum_vec = _mm256_fmadd_ps(p1, k_vec_1, sum_vec);
            // sum_vec = _mm256_fmadd_ps(p2, k_vec_2, sum_vec);
            // sum_vec = _mm256_fmadd_ps(p3, k_vec_3, sum_vec);
            // sum_vec = _mm256_fmadd_ps(p4, k_vec_4, sum_vec);
            // sum_vec = _mm256_fmadd_ps(p5, k_vec_5, sum_vec);
            // sum_vec = _mm256_fmadd_ps(p6, k_vec_6, sum_vec);
            // sum_vec = _mm256_fmadd_ps(p7, k_vec_7, sum_vec);
            // sum_vec = _mm256_fmadd_ps(p8, k_vec_8, sum_vec);

            // --- 4. Normalize and Store ---
            sum_vec = _mm256_div_ps(sum_vec, k_vec_norm);
            _mm256_storeu_ps(&p_dst[x], sum_vec);
        }
        
        // --- 5. Remainder Loop (Serial) ---
        // Handle remaining pixels (less than 8) at the end of the row
        for (; x < cols - 1; ++x) {
            float sum = 0.0f;
            sum += p_top[x - 1] * kernel[0];
            sum += p_top[x]     * kernel[1];
            sum += p_top[x + 1] * kernel[2];
            sum += p_mid[x - 1] * kernel[3];
            sum += p_mid[x]     * kernel[4];
            sum += p_mid[x + 1] * kernel[5];
            sum += p_bot[x - 1] * kernel[6];
            sum += p_bot[x]     * kernel[7];
            sum += p_bot[x + 1] * kernel[8];
            p_dst[x] = sum / norm_factor;
        }
    }
}


/**
 * @brief Fused Serial implementation of SobelX, SobelY, and Magnitude
 */
void sobel_magnitude_fused_serial(const Mat& blurred, Mat& magnitude) {
    int rows = blurred.rows;
    int cols = blurred.cols;
    magnitude.create(rows, cols, CV_32F);
    
    // Set borders to zero
    magnitude.setTo(Scalar(0.0));

    for (int y = 1; y < rows - 1; ++y) {
        const float* p_top = blurred.ptr<float>(y - 1);
        const float* p_mid = blurred.ptr<float>(y);
        const float* p_bot = blurred.ptr<float>(y + 1);
        float* p_mag = magnitude.ptr<float>(y);

        for (int x = 1; x < cols - 1; ++x) {
            float gx = (p_top[x - 1] * sobel_x_kernel[0]) + (p_top[x + 1] * sobel_x_kernel[2]) +
                       (p_mid[x - 1] * sobel_x_kernel[3]) + (p_mid[x + 1] * sobel_x_kernel[5]) +
                       (p_bot[x - 1] * sobel_x_kernel[6]) + (p_bot[x + 1] * sobel_x_kernel[8]);
            
            float gy = (p_top[x - 1] * sobel_y_kernel[0]) + (p_top[x] * sobel_y_kernel[1]) + (p_top[x + 1] * sobel_y_kernel[2]) +
                       (p_bot[x - 1] * sobel_y_kernel[6]) + (p_bot[x] * sobel_y_kernel[7]) + (p_bot[x + 1] * sobel_y_kernel[8]);
            
            p_mag[x] = sqrt(gx * gx + gy * gy);
        }
    }
}

/**
 * @brief Fused Parallel implementation of SobelX, SobelY, and Magnitude
 */
void sobel_magnitude_fused_parallel(const Mat& blurred, Mat& magnitude) {
    int rows = blurred.rows;
    int cols = blurred.cols;
    magnitude.create(rows, cols, CV_32F);
    
    // Set borders to zero
    magnitude.setTo(Scalar(0.0));

    // --- 1. Prepare Sobel X Kernel Registers (only non-zero weights) ---
    const __m256 sx_vec_0 = _mm256_set1_ps(sobel_x_kernel[0]); // -1.0
    const __m256 sx_vec_2 = _mm256_set1_ps(sobel_x_kernel[2]); //  1.0
    const __m256 sx_vec_3 = _mm256_set1_ps(sobel_x_kernel[3]); // -2.0
    const __m256 sx_vec_5 = _mm256_set1_ps(sobel_x_kernel[5]); //  2.0
    const __m256 sx_vec_6 = _mm256_set1_ps(sobel_x_kernel[6]); // -1.0
    const __m256 sx_vec_8 = _mm256_set1_ps(sobel_x_kernel[8]); //  1.0

    // --- 2. Prepare Sobel Y Kernel Registers (only non-zero weights) ---
    const __m256 sy_vec_0 = _mm256_set1_ps(sobel_y_kernel[0]); // -1.0
    const __m256 sy_vec_1 = _mm256_set1_ps(sobel_y_kernel[1]); // -2.0
    const __m256 sy_vec_2 = _mm256_set1_ps(sobel_y_kernel[2]); // -1.0
    const __m256 sy_vec_6 = _mm256_set1_ps(sobel_y_kernel[6]); //  1.0
    const __m256 sy_vec_7 = _mm256_set1_ps(sobel_y_kernel[7]); //  2.0
    const __m256 sy_vec_8 = _mm256_set1_ps(sobel_y_kernel[8]); //  1.0

    const int limit = cols - 9; // (cols - 1) - 8

    for (int y = 1; y < rows - 1; ++y) {
        const float* p_top = blurred.ptr<float>(y - 1);
        const float* p_mid = blurred.ptr<float>(y);
        const float* p_bot = blurred.ptr<float>(y + 1);
        float* p_mag = magnitude.ptr<float>(y);
        
        int x = 1;

        // --- 3. Parallel Fused Loop ---
        for (; x <= limit; x += 8) {
            // Load all 9 neighbor pixel blocks
            const __m256 p0 = _mm256_loadu_ps(&p_top[x - 1]);
            const __m256 p1 = _mm256_loadu_ps(&p_top[x]);
            const __m256 p2 = _mm256_loadu_ps(&p_top[x + 1]);
            const __m256 p3 = _mm256_loadu_ps(&p_mid[x - 1]);
            //const __m256 p4 = _mm256_loadu_ps(&p_mid[x]); // Not used by Sobel X
            const __m256 p5 = _mm256_loadu_ps(&p_mid[x + 1]);
            const __m256 p6 = _mm256_loadu_ps(&p_bot[x - 1]);
            const __m256 p7 = _mm256_loadu_ps(&p_bot[x]);
            const __m256 p8 = _mm256_loadu_ps(&p_bot[x + 1]);

            // --- 4. Calculate Sobel X for 8 pixels (Old Method) ---
            __m256 gx_vec = _mm256_setzero_ps();
            gx_vec = _mm256_add_ps(gx_vec, _mm256_mul_ps(p0, sx_vec_0));
            gx_vec = _mm256_add_ps(gx_vec, _mm256_mul_ps(p2, sx_vec_2));
            gx_vec = _mm256_add_ps(gx_vec, _mm256_mul_ps(p3, sx_vec_3));
            gx_vec = _mm256_add_ps(gx_vec, _mm256_mul_ps(p5, sx_vec_5));
            gx_vec = _mm256_add_ps(gx_vec, _mm256_mul_ps(p6, sx_vec_6));
            gx_vec = _mm256_add_ps(gx_vec, _mm256_mul_ps(p8, sx_vec_8));
            
            // --- (Commented) Sobel X (Newer Method) ---
            // __m256 gx_vec = _mm256_setzero_ps();
            // gx_vec = _mm256_fmadd_ps(p0, sx_vec_0, gx_vec);
            // gx_vec = _mm256_fmadd_ps(p2, sx_vec_2, gx_vec);
            // gx_vec = _mm256_fmadd_ps(p3, sx_vec_3, gx_vec);
            // gx_vec = _mm256_fmadd_ps(p5, sx_vec_5, gx_vec);
            // gx_vec = _mm256_fmadd_ps(p6, sx_vec_6, gx_vec);
            // gx_vec = _mm256_fmadd_ps(p8, sx_vec_8, gx_vec);

            
            // --- 5. Calculate Sobel Y for 8 pixels (Old Method) ---
            __m256 gy_vec = _mm256_setzero_ps();
            gy_vec = _mm256_add_ps(gy_vec, _mm256_mul_ps(p0, sy_vec_0));
            gy_vec = _mm256_add_ps(gy_vec, _mm256_mul_ps(p1, sy_vec_1));
            gy_vec = _mm256_add_ps(gy_vec, _mm256_mul_ps(p2, sy_vec_2));
            gy_vec = _mm256_add_ps(gy_vec, _mm256_mul_ps(p6, sy_vec_6));
            gy_vec = _mm256_add_ps(gy_vec, _mm256_mul_ps(p7, sy_vec_7));
            gy_vec = _mm256_add_ps(gy_vec, _mm256_mul_ps(p8, sy_vec_8));

            // --- (Commented) Sobel Y (Newer Method) ---
            // __m256 gy_vec = _mm256_setzero_ps();
            // gy_vec = _mm256_fmadd_ps(p0, sy_vec_0, gy_vec);
            // gy_vec = _mm256_fmadd_ps(p1, sy_vec_1, gy_vec);
            // gy_vec = _mm256_fmadd_ps(p2, sy_vec_2, gy_vec);
            // gy_vec = _mm256_fmadd_ps(p6, sy_vec_6, gy_vec);
            // gy_vec = _mm256_fmadd_ps(p7, sy_vec_7, gy_vec);
            // gy_vec = _mm256_fmadd_ps(p8, sy_vec_8, gy_vec);

            // --- 6. Calculate Magnitude for 8 pixels ---
            // mag = sqrt(gx*gx + gy*gy)
            __m256 gx_sq = _mm256_mul_ps(gx_vec, gx_vec);
            __m256 gy_sq = _mm256_mul_ps(gy_vec, gy_vec);
            __m256 sum_sq = _mm256_add_ps(gx_sq, gy_sq);
            __m256 mag_vec = _mm256_sqrt_ps(sum_sq);

            // Store 8 results
            _mm256_storeu_ps(&p_mag[x], mag_vec);
        }
        
        // --- 7. Remainder Loop (Serial) ---
        for (; x < cols - 1; ++x) {
            float gx = (p_top[x - 1] * sobel_x_kernel[0]) + (p_top[x + 1] * sobel_x_kernel[2]) +
                       (p_mid[x - 1] * sobel_x_kernel[3]) + (p_mid[x + 1] * sobel_x_kernel[5]) +
                       (p_bot[x - 1] * sobel_x_kernel[6]) + (p_bot[x + 1] * sobel_x_kernel[8]);
            
            float gy = (p_top[x - 1] * sobel_y_kernel[0]) + (p_top[x] * sobel_y_kernel[1]) + (p_top[x + 1] * sobel_y_kernel[2]) +
                       (p_bot[x - 1] * sobel_y_kernel[6]) + (p_bot[x] * sobel_y_kernel[7]) + (p_bot[x + 1] * sobel_y_kernel[8]);
            
            p_mag[x] = sqrt(gx * gx + gy * gy);
        }
    }
}


int main() {
    // --- Step 0: Load and Prepare Image ---
    Mat base_img = imread("../assets/base.jpg");
    if (base_img.empty()) {
        cout << "Error: base.jpg image not found. Make sure it's in the same directory." << endl;
        return -1;
    }

    Mat gray_img;
    cvtColor(base_img, gray_img, COLOR_BGR2GRAY);

    Mat gray_float;
    gray_img.convertTo(gray_float, CV_32F); // Convert to float for calculations

    // --- Prepare Serial Output ---
    Mat blurred_serial(gray_float.size(), CV_32F);
    Mat magnitude_serial(gray_float.size(), CV_32F);
    Mat magnitude_serial_8u(gray_float.size(), CV_8U); // Final 8-bit output
    
    // --- Prepare Parallel Output ---
    Mat blurred_parallel(gray_float.size(), CV_32F);
    Mat magnitude_parallel(gray_float.size(), CV_32F);
    Mat magnitude_parallel_8u(gray_float.size(), CV_8U); // Final 8-bit output

    cout << "Image Size: " << gray_img.cols << "x" << gray_img.rows << endl;
    
    // --- Timing (Serial) ---
    cout << "--- Running *Optimized* Serial Sobel Pipeline ---" << endl;
    unsigned long long start_serial, end_serial;
    start_serial = __rdtsc();

    // Step 1: Gaussian
    convolve_serial(gray_float, blurred_serial, gaussian_kernel, GAUSSIAN_NORM);
    // Step 2, 3, 4: Fused Sobel + Magnitude
    sobel_magnitude_fused_serial(blurred_serial, magnitude_serial);

    end_serial = __rdtsc();
    unsigned long long time_serial = end_serial - start_serial;
    cout << "Optimized Serial Time (clocks): " << time_serial << endl;
    
    // Save serial result
    magnitude_serial.convertTo(magnitude_serial_8u, CV_8U);
    imwrite("sobel_serial_result.jpg", magnitude_serial_8u);
    cout << "Serial result saved to sobel_serial_result.jpg" << endl;


    // --- Timing (Parallel) ---
    cout << "--- Running Parallel Sobel Pipeline ---" << endl;
    unsigned long long start_parallel, end_parallel;
    start_parallel = __rdtsc();

    // Step 1: Parallel Gaussian
    convolve_parallel(gray_float, blurred_parallel, gaussian_kernel, GAUSSIAN_NORM);
    // Step 2, 3, 4: Parallel Fused Sobel + Magnitude
    sobel_magnitude_fused_parallel(blurred_parallel, magnitude_parallel);
    
    end_parallel = __rdtsc();
    unsigned long long time_parallel = end_parallel - start_parallel;
    cout << "Parallel Time (clocks): " << time_parallel << endl;

    // Save parallel result
    magnitude_parallel.convertTo(magnitude_parallel_8u, CV_8U);
    imwrite("sobel_parallel_result.jpg", magnitude_parallel_8u);
    cout << "Parallel result saved to sobel_parallel_result.jpg" << endl;

    // --- Final Speedup ---
    cout << "--- Results ---" << endl;
    cout << "Serial Time:   " << time_serial << " clocks" << endl;
    cout << "Parallel Time: " << time_parallel << " clocks" << endl;
    cout << "Speedup:       " << std::fixed << std::setprecision(2) << (double)time_serial / time_parallel << "x" << endl;

    return 0;
}

