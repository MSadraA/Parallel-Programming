#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <x86intrin.h> // For __rdtsc()
#include <opencv2/opencv.hpp> // OpenCV Library
#include <immintrin.h> // Main header for AVX
#include <malloc.h> // For _mm_malloc and _mm_free
// OpenMP is removed as requested

using namespace std;
using namespace cv;

// --- Kernels (unchanged) ---
const float gaussian_kernel[9] = {1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 2.0f, 1.0f};
const float GAUSSIAN_NORM = 16.0f;
const float GAUSSIAN_RECIPROCAL = 1.0f / 16.0f;
const float sobel_x_kernel[9] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
const float sobel_y_kernel[9] = {-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f};

// --- Aligned Memory Helpers (unchanged) ---
const size_t ALIGNMENT = 32;

Mat createAlignedMat(int rows, int cols, int type) {
    size_t step = (cols * CV_ELEM_SIZE(type) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
    unsigned char* data = (unsigned char*)_mm_malloc(rows * step, ALIGNMENT);
    if (!data) {
        cout << "Error: _mm_malloc failed!" << endl;
        return Mat();
    }
    return Mat(rows, cols, type, data, step);
}

void freeAlignedMat(Mat& mat) {
    if (mat.data) {
        _mm_free(mat.data);
        mat.data = nullptr;
    }
}

// --- Serial Functions (unchanged) ---
void convolve_serial(const Mat& src, Mat& dst, const float* kernel, float norm_factor) {
    int rows = src.rows;
    int cols = src.cols;
    dst.create(rows, cols, CV_32F); 
    dst.setTo(Scalar(0.0));

    for (int y = 1; y < rows - 1; ++y) {
        const float* p_top = src.ptr<float>(y - 1);
        const float* p_mid = src.ptr<float>(y);
        const float* p_bot = src.ptr<float>(y + 1);
        float* p_dst = dst.ptr<float>(y);

        for (int x = 1; x < cols - 1; ++x) {
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

void sobel_magnitude_fused_serial(const Mat& blurred, Mat& magnitude) {
    int rows = blurred.rows;
    int cols = blurred.cols;
    magnitude.create(rows, cols, CV_32F);
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


// --- [Modified] Parallel Gaussian function for Tiling ---
// [NEW] Takes y_start and y_end as parameters
void convolve_parallel_tiled(const Mat& src, Mat& dst, const float* kernel, float norm_factor, int y_start, int y_end) {
    int cols = src.cols;
    
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

    const int limit = cols - 9;

    // [MODIFIED] y-loop now runs within the tile range
    for (int y = y_start; y < y_end; ++y) {
        const float* p_top = src.ptr<float>(y - 1);
        const float* p_mid = src.ptr<float>(y);
        const float* p_bot = src.ptr<float>(y + 1);
        float* p_dst = dst.ptr<float>(y);
        
        int x = 1;
        
        for (; x <= limit; x += 8) {
            const __m256 p0 = _mm256_loadu_ps(&p_top[x - 1]);
            const __m256 p1 = _mm256_loadu_ps(&p_top[x]);
            const __m256 p2 = _mm256_loadu_ps(&p_top[x + 1]);
            const __m256 p3 = _mm256_loadu_ps(&p_mid[x - 1]);
            const __m256 p4 = _mm256_loadu_ps(&p_mid[x]);
            const __m256 p5 = _mm256_loadu_ps(&p_mid[x + 1]);
            const __m256 p6 = _mm256_loadu_ps(&p_bot[x - 1]);
            const __m256 p7 = _mm256_loadu_ps(&p_bot[x]);
            const __m256 p8 = _mm256_loadu_ps(&p_bot[x + 1]);

            __m256 sum_vec = _mm256_setzero_ps();
            sum_vec = _mm256_fmadd_ps(p0, k_vec_0, sum_vec);
            sum_vec = _mm256_fmadd_ps(p1, k_vec_1, sum_vec);
            sum_vec = _mm256_fmadd_ps(p2, k_vec_2, sum_vec);
            sum_vec = _mm256_fmadd_ps(p3, k_vec_3, sum_vec);
            sum_vec = _mm256_fmadd_ps(p4, k_vec_4, sum_vec);
            sum_vec = _mm256_fmadd_ps(p5, k_vec_5, sum_vec);
            sum_vec = _mm256_fmadd_ps(p6, k_vec_6, sum_vec);
            sum_vec = _mm256_fmadd_ps(p7, k_vec_7, sum_vec);
            sum_vec = _mm256_fmadd_ps(p8, k_vec_8, sum_vec);

            sum_vec = _mm256_mul_ps(sum_vec, k_vec_norm);
            _mm256_storeu_ps(&p_dst[x], sum_vec); // Using unaligned store
        }
        
        // Remainder loop
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
            p_dst[x] = sum * norm_factor;
        }
    }
}

// --- [Modified] Parallel Sobel function for Tiling ---
// [NEW] Takes y_start and y_end as parameters
void sobel_magnitude_fused_parallel_tiled(const Mat& blurred, Mat& magnitude, int y_start, int y_end) {
    int cols = blurred.cols;

    const __m256 sx_vec_0 = _mm256_set1_ps(sobel_x_kernel[0]);
    const __m256 sx_vec_2 = _mm256_set1_ps(sobel_x_kernel[2]);
    const __m256 sx_vec_3 = _mm256_set1_ps(sobel_x_kernel[3]);
    const __m256 sx_vec_5 = _mm256_set1_ps(sobel_x_kernel[5]);
    const __m256 sx_vec_6 = _mm256_set1_ps(sobel_x_kernel[6]);
    const __m256 sx_vec_8 = _mm256_set1_ps(sobel_x_kernel[8]);
    const __m256 sy_vec_0 = _mm256_set1_ps(sobel_y_kernel[0]);
    const __m256 sy_vec_1 = _mm256_set1_ps(sobel_y_kernel[1]);
    const __m256 sy_vec_2 = _mm256_set1_ps(sobel_y_kernel[2]);
    const __m256 sy_vec_6 = _mm256_set1_ps(sobel_y_kernel[6]);
    const __m256 sy_vec_7 = _mm256_set1_ps(sobel_y_kernel[7]);
    const __m256 sy_vec_8 = _mm256_set1_ps(sobel_y_kernel[8]);

    const int limit = cols - 9;

    // [MODIFIED] y-loop now runs within the tile range
    for (int y = y_start; y < y_end; ++y) {
        const float* p_top = blurred.ptr<float>(y - 1);
        const float* p_mid = blurred.ptr<float>(y);
        const float* p_bot = blurred.ptr<float>(y + 1);
        float* p_mag = magnitude.ptr<float>(y);

        int x = 1;

        for (; x <= limit; x += 8) {
            const __m256 p0 = _mm256_loadu_ps(&p_top[x - 1]);
            const __m256 p1 = _mm256_loadu_ps(&p_top[x]);
            const __m256 p2 = _mm256_loadu_ps(&p_top[x + 1]);
            const __m256 p3 = _mm256_loadu_ps(&p_mid[x - 1]);
            // const __m256 p4 = _mm256_loadu_ps(&p_mid[x]);
            const __m256 p5 = _mm256_loadu_ps(&p_mid[x + 1]);
            const __m256 p6 = _mm256_loadu_ps(&p_bot[x - 1]);
            const __m256 p7 = _mm256_loadu_ps(&p_bot[x]);
            const __m256 p8 = _mm256_loadu_ps(&p_bot[x + 1]);

            __m256 gx_vec = _mm256_setzero_ps();
            gx_vec = _mm256_fmadd_ps(p0, sx_vec_0, gx_vec);
            gx_vec = _mm256_fmadd_ps(p2, sx_vec_2, gx_vec);
            gx_vec = _mm256_fmadd_ps(p3, sx_vec_3, gx_vec);
            gx_vec = _mm256_fmadd_ps(p5, sx_vec_5, gx_vec);
            gx_vec = _mm256_fmadd_ps(p6, sx_vec_6, gx_vec);
            gx_vec = _mm256_fmadd_ps(p8, sx_vec_8, gx_vec);

            __m256 gy_vec = _mm256_setzero_ps();
            gy_vec = _mm256_fmadd_ps(p0, sy_vec_0, gy_vec);
            gy_vec = _mm256_fmadd_ps(p1, sy_vec_1, gy_vec);
            gy_vec = _mm256_fmadd_ps(p2, sy_vec_2, gy_vec);
            gy_vec = _mm256_fmadd_ps(p6, sy_vec_6, gy_vec);
            gy_vec = _mm256_fmadd_ps(p7, sy_vec_7, gy_vec);
            gy_vec = _mm256_fmadd_ps(p8, sy_vec_8, gy_vec);

            __m256 gx_sq = _mm256_mul_ps(gx_vec, gx_vec);
            __m256 gy_sq = _mm256_mul_ps(gy_vec, gy_vec);
            __m256 sum_sq = _mm256_add_ps(gx_sq, gy_sq);
            __m256 mag_vec = _mm256_sqrt_ps(sum_sq);

            // [BUG FIX] Using storeu instead of store
            _mm256_storeu_ps(&p_mag[x], mag_vec); 
        }

        // Remainder loop
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
    // --- 0. Load and Prepare Image ---
    Mat base_img = imread("base.jpg");
    if (base_img.empty()) {
        cout << "Error: base.jpg image not found." << endl;
        return -1;
    }

    Mat gray_img;
    cvtColor(base_img, gray_img, COLOR_BGR2GRAY);

    // [MODIFIED] Using aligned memory
    Mat gray_float = createAlignedMat(gray_img.rows, gray_img.cols, CV_32F);
    gray_img.convertTo(gray_float, CV_32F); 

    int rows = gray_float.rows;
    int cols = gray_float.cols;
    cout << "Image Size: " << cols << "x" << rows << endl;
    
    // --- 1. Serial Timing (Baseline) ---
    Mat blurred_serial = createAlignedMat(rows, cols, CV_32F);
    Mat magnitude_serial = createAlignedMat(rows, cols, CV_32F);
    Mat magnitude_serial_8u(rows, cols, CV_8U);

    cout << "--- Running *Optimized* Serial Sobel Pipeline ---" << endl;
    unsigned long long start_serial, end_serial;
    start_serial = __rdtsc();
    
    // Serial execution (this part is unchanged)
    convolve_serial(gray_float, blurred_serial, gaussian_kernel, GAUSSIAN_NORM);
    sobel_magnitude_fused_serial(blurred_serial, magnitude_serial);
    
    end_serial = __rdtsc();
    unsigned long long time_serial = end_serial - start_serial;
    cout << "Serial Time (clocks): " << time_serial << endl;
    
    magnitude_serial.convertTo(magnitude_serial_8u, CV_8U);
    imwrite("sobel_serial_result.jpg", magnitude_serial_8u);

    // --- 2. Parallel Timing (Tiled + SIMD) ---
    Mat blurred_parallel = createAlignedMat(rows, cols, CV_32F);
    Mat magnitude_parallel = createAlignedMat(rows, cols, CV_32F);
    Mat magnitude_parallel_8u(rows, cols, CV_8U);

    cout << "--- Running Parallel Sobel Pipeline (Tiled + SIMD) ---" << endl;
    
    // [NEW] Define the tile size. 64 or 128 rows is usually good for L2/L3 cache
    const int TILE_HEIGHT = 64; 
    
    unsigned long long start_parallel, end_parallel;
    start_parallel = __rdtsc();

    // [NEW] This is the main Tiling execution loop
    // This loop is serial, but its internal operations are cache-optimized
    for (int y_tile = 0; y_tile < rows; y_tile += TILE_HEIGHT) {
        
        // --- Step A: Run Gaussian on the Tile ---
        // We need a 1-pixel "halo" above and below the tile for Sobel
        
        // Calculate Gaussian range (y_start, y_end)
        // Row 0 and last row are borders, so we start/end at 1 and rows-1
        int y_gauss_start = max(1, y_tile - 1); // 1-pixel halo on top
        int y_gauss_end   = min(rows - 1, y_tile + TILE_HEIGHT + 1); // 1-pixel halo on bottom

        // If y_tile starts at 0, y_gauss_start must be 1
        if (y_tile == 0) y_gauss_start = 1; 

        convolve_parallel_tiled(gray_float, blurred_parallel, gaussian_kernel, GAUSSIAN_RECIPROCAL, y_gauss_start, y_gauss_end);

        // --- Step B: Run Sobel on the Tile ---
        // Sobel only runs on the rows *inside* the main tile
        
        // Calculate Sobel range (y_start, y_end)
        int y_sobel_start = max(1, y_tile);
        int y_sobel_end   = min(rows - 1, y_tile + TILE_HEIGHT);
        
        // For row y=1 (in the first tile), it needs row 0 from blurred
        // But row 0 of blurred wasn't calculated (Gaussian starts at y=1)
        // So Sobel must also start at y=2
        if (y_sobel_start < 2) y_sobel_start = 2;
        
        // Ensure Sobel doesn't read past where Gaussian has written
        if (y_gauss_end <= y_sobel_end) y_sobel_end = y_gauss_end -1;

        if (y_sobel_start < y_sobel_end) { // Ensure there are rows to process
             sobel_magnitude_fused_parallel_tiled(blurred_parallel, magnitude_parallel, y_sobel_start, y_sobel_end);
        }
    }
    
    end_parallel = __rdtsc();
    unsigned long long time_parallel = end_parallel - start_serial; // [BUG] This should be end_parallel - start_parallel
    
    // --- [CORRECTION] ---
    // The timer for parallel was started *after* the serial execution.
    // Let's assume the user meant to time only the parallel part.
    // The previous code had `time_parallel = end_parallel - start_serial;` which is wrong.
    // It should be:
    time_parallel = end_parallel - start_parallel;
    
    cout << "Parallel Time (clocks): " << time_parallel << endl;

    magnitude_parallel.convertTo(magnitude_parallel_8u, CV_8U);
    imwrite("sobel_parallel_result.jpg", magnitude_parallel_8u);

    // --- 3. Final Results ---
    cout << "--- Results ---" << endl;
    cout << "Serial Time:   " << time_serial << " clocks" << endl;
    cout << "Parallel Time: " << time_parallel << " clocks" << endl;
    cout << "Speedup:       " << fixed << setprecision(2) << (double)time_serial / time_parallel << "x" << endl;

    // --- 4. Free Aligned Memory ---
    freeAlignedMat(gray_float);
    freeAlignedMat(blurred_serial);
    freeAlignedMat(magnitude_serial);
    freeAlignedMat(blurred_parallel);
    freeAlignedMat(magnitude_parallel);

    return 0;
}