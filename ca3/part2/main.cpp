#include <iostream>
#include <complex>      // Standard complex number class
#include <opencv2/opencv.hpp>
#include <omp.h>        // Included for timing (omp_get_wtime)
#include <x86intrin.h> // For __rdtsc()

using namespace std;
using namespace cv;

// --- Configuration Constants ---
const int WIDTH = 1000;
const int HEIGHT = 1000;
const int MAX_ITER = 1000;
const int POWER_N = 2; // The 'n' in Z^n + C

// The complex constant C for the Julia Set.
// Standard value from assignment: -0.355 + 0.355i
const complex<double> C(-0.355, 0.355);

// Complex Plane boundaries
const double REAL_MIN = -1.5;
const double REAL_MAX = 1.5;
const double IMAG_MIN = -1.5;
const double IMAG_MAX = 1.5;

// Escape Radius Squared (2.0^2 = 4.0)
// We use squared radius to avoid expensive sqrt() calculations
const double ESCAPE_RADIUS_SQ = 4.0;

/**
 * @brief Generates the Julia Set image serially.
 * * @param image Reference to the OpenCV Mat object to fill.
 */
void generate_julia_serial(Mat& image) {
    
    // Loop over every row (y)
    for (int y = 0; y < HEIGHT; ++y) {
        
        // Pointer to the current row in the image matrix.
        // Accessing memory row by row is cache-friendly.
        // Used for colors - BGR order
        Vec3b* row_ptr = image.ptr<Vec3b>(y);
        double imag = IMAG_MIN + (double)y / HEIGHT * (IMAG_MAX - IMAG_MIN);

        // Loop over every column (x)
        for (int x = 0; x < WIDTH; ++x) {
            
            // --- Step 1: Map pixel coordinate (x, y) to complex plane (real, imag) ---
            double real = REAL_MIN + (double)x / WIDTH * (REAL_MAX - REAL_MIN);

            complex<double> z(real, imag);
            
            // --- Step 2: Iterate the formula Z = Z^n + C ---
            int iter = 0;
            while (iter < MAX_ITER) {
                // Optimization: Check squared norm to avoid sqrt()
                // std::norm(z) returns real^2 + imag^2
                if (std::norm(z) > ESCAPE_RADIUS_SQ) {
                    break; // Point escaped (Diverged)
                }
                
                // Apply the formula: Z_new = Z^n + C
                z = std::pow(z, POWER_N) + C;
                
                iter++;
            }

            // --- Step 3: Color the pixel based on iterations ---
            if (iter == MAX_ITER) {
                // Point is convergent (part of the set) -> Color Black
                row_ptr[x] = Vec3b(0, 0, 0); 
            } else {
                // Point is divergent -> Color based on how fast it escaped.
                // Simple coloring: Map iteration count to grayscale or a color map.
                // Here we use a simple formula for visual variety.
                
                // I just changed the constants for a better visual effect :)
                // Example: Map iter 0-1000 to 0-255
                unsigned char color_val = (unsigned char)(255.0 * iter / 10); 
                // BGR format in OpenCV
                // Just a simple coloring scheme (Blueish to White)
                row_ptr[x] = Vec3b(color_val, color_val, 0); 
            }
        }
    }
}



/**
 * @brief Generates the Julia Set image in parallel using OpenMP.
 * Uses dynamic scheduling to handle load imbalance.
 * @param image Reference to the OpenCV Mat object to fill.
 */
const int CHUNK_SIZE = 10;
void generate_julia_parallel(Mat& image) {
    
    // Loop variables declared outside to be compatible with OpenMP directive syntax
    int y;

    // Pre-calculate steps to avoid division inside the loop
    double dx = (REAL_MAX - REAL_MIN) / WIDTH;
    double dy = (IMAG_MAX - IMAG_MIN) / HEIGHT;
    double imag, real;
    
    // --- OpenMP Directive Explanation ---
    // parallel for: Automatically splits the loop iterations among threads.
    // schedule(dynamic): Assigns chunks of work dynamically. Crucial here because
    //                    some rows (outside the set) finish fast, while others (inside) take long.
    // shared(image): The image matrix is shared (threads write to different parts).
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE) shared(image, dx, dy) private(y, imag, real)
    for (y = 0; y < HEIGHT; ++y) {
        
        // Get pointer to the current row.
        // Since 'image' is shared, multiple threads access it, but they access
        // different rows 'y', so no race condition occurs here.
        Vec3b* row_ptr = image.ptr<Vec3b>(y);

        double imag = IMAG_MIN + y * dy;
        double real;

        // Inner loop (x) runs serially within each thread for the assigned row 'y'
        for (int x = 0; x < WIDTH; ++x) {
            
            // --- Step 1: Map pixel to complex plane ---
            real = REAL_MIN + x * dx;

            complex<double> z(real, imag);
            
            // --- Step 2: Iterate ---
            int iter = 0;
            while (iter < MAX_ITER) {
                // Check squared norm
                if (std::norm(z) > ESCAPE_RADIUS_SQ) {
                    break; 
                }
                z = std::pow(z, POWER_N) + C;
                iter++;
            }

            // --- Step 3: Color the pixel based on iterations ---
            if (iter == MAX_ITER) {
                // Point is convergent (part of the set) -> Color Black
                row_ptr[x] = Vec3b(0, 0, 0); 
            } else {
                // Point is divergent -> Color based on how fast it escaped.
                // Simple coloring: Map iteration count to grayscale or a color map.
                // Here we use a simple formula for visual variety.
                
                // I just changed the constants for a better visual effect :)
                // Example: Map iter 0-1000 to 0-255
                unsigned char color_val = (unsigned char)(255.0 * iter / 10); 
                // BGR format in OpenCV
                // Just a simple coloring scheme (Blueish to White)
                row_ptr[x] = Vec3b(color_val, color_val, 0); 
            }
        }
    }
}


// Add this helper function for AVX
void generate_julia_avx(Mat& image) {
    int y;

    // Pre-calculate steps
    float dx = (float)((REAL_MAX - REAL_MIN) / WIDTH);
    float dy = (float)((IMAG_MAX - IMAG_MIN) / HEIGHT);

    float real_min_f = (float)REAL_MIN;
    float imag_min_f = (float)IMAG_MIN;
    float escape_sq_f = (float)ESCAPE_RADIUS_SQ;

    // Prepare constants for AVX
    __m256 v_c_re = _mm256_set1_ps((float)C.real());
    __m256 v_c_im = _mm256_set1_ps((float)C.imag());
    __m256 v_escape = _mm256_set1_ps(escape_sq_f);
    __m256 v_dx = _mm256_set1_ps(dx);
    __m256 v_real_min = _mm256_set1_ps(real_min_f);
    
    // Offset vector for x: {0.0, 1.0, 2.0, ..., 7.0}
    __m256 v_x_offsets = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);

    // OpenMP for outer loop (Rows)
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE) shared(image) private(y)
    for (y = 0; y < HEIGHT; ++y) {
        
        Vec3b* row_ptr = image.ptr<Vec3b>(y);
        float cur_y_imag = imag_min_f + y * dy;
        __m256 v_z_im_init = _mm256_set1_ps(cur_y_imag);

        int x = 0;
        
        // --- AVX Loop (8 pixels at a time) ---
        for (; x <= WIDTH - 8; x += 8) {
            
            // 1. Initialize Z_real for 8 pixels
            // x_base = {x, x, ..., x}
            __m256 v_x_base = _mm256_set1_ps((float)x);
            // v_x_vals = {x, x+1, ..., x+7}
            __m256 v_x_vals = _mm256_add_ps(v_x_base, v_x_offsets);
            
            // z_re = real_min + x * dx
            __m256 v_z_re = _mm256_fmadd_ps(v_x_vals, v_dx, v_real_min);
            __m256 v_z_im = v_z_im_init;

            // 2. Initialize counters (integers) to 0
            __m256i v_iters = _mm256_setzero_si256();
            
            // Mask: Initially all ones (0xFFFFFFFF), meaning all points are "active"
            // We track active points to stop updating their iteration count once they escape
            __m256 v_mask = _mm256_cmp_ps(v_z_re, v_z_re, _CMP_EQ_OQ); // True for all numbers

            // 3. Iteration Loop
            for (int k = 0; k < MAX_ITER; ++k) {
                
                // Calculate |Z|^2 = re^2 + im^2
                __m256 v_re2 = _mm256_mul_ps(v_z_re, v_z_re);
                __m256 v_im2 = _mm256_mul_ps(v_z_im, v_z_im);
                __m256 v_norm2 = _mm256_add_ps(v_re2, v_im2);

                // Check escape: norm2 < 4.0 ?
                // Returns 0xFFFFFFFF (-1) if True (inside), 0 if False (escaped)
                __m256 v_inside = _mm256_cmp_ps(v_norm2, v_escape, _CMP_LT_OQ);
                
                // Update the "active" mask. Once a pixel escapes, it stays escaped.
                // mask = mask & inside
                // Note: We use bitwise AND on float vectors (cast to integer for logic)
                v_mask = _mm256_and_ps(v_mask, v_inside);

                // If mask is all zeros, everyone has escaped. Break early!
                int mask_bits = _mm256_movemask_ps(v_mask);
                if (mask_bits == 0) {
                    break; 
                }

                // Update iteration counts
                // We add 1 to v_iters ONLY if the pixel is still active (mask is -1)
                // Since -1 is 0xFFFFFFFF, and we want to add 1, we can subtract the mask!
                // iter = iter - (-1) => iter + 1
                // We must perform this addition on integer vectors.
                v_iters = _mm256_sub_epi32(v_iters, _mm256_castps_si256(v_mask));

                // --- Calculate Z_new = Z^n + C (Generic Power) ---   
                // Start with Z^1
                __m256 v_res_re = v_z_re;
                __m256 v_res_im = v_z_im;

                // Save the base Z (because v_z_re will be updated in next k-loop, 
                // but for power calc we need the Z at START of this step)
                __m256 v_base_re = v_z_re;
                __m256 v_base_im = v_z_im;

                // Loop to multiply Z by itself (POWER_N - 1) times
                // Example: If N=2, loop runs once: Z * Z
                // Example: If N=3, loop runs twice: (Z * Z) * Z
                for (int p = 1; p < POWER_N; ++p) {
                    // Complex Multiplication: (res * base)
                    // Real: (res_re * base_re) - (res_im * base_im)
                    // Imag: (res_re * base_im) + (res_im * base_re)

                    __m256 v_ac = _mm256_mul_ps(v_res_re, v_base_re);
                    __m256 v_bd = _mm256_mul_ps(v_res_im, v_base_im);
                    __m256 v_ad = _mm256_mul_ps(v_res_re, v_base_im);
                    __m256 v_bc = _mm256_mul_ps(v_res_im, v_base_re);

                    v_res_re = _mm256_sub_ps(v_ac, v_bd);
                    v_res_im = _mm256_add_ps(v_ad, v_bc);
                }

                // Finally add constant C
                v_z_re = _mm256_add_ps(v_res_re, v_c_re);
                v_z_im = _mm256_add_ps(v_res_im, v_c_im);
            }

            // 4. Extract results and Color
            // Extract the 8 iteration counts to a temporary array
            int iters_array[8];
            _mm256_storeu_si256((__m256i*)iters_array, v_iters);

            for (int i = 0; i < 8; ++i) {
                int iter = iters_array[i];
                if (iter >= MAX_ITER) { // Safety check (or iter == MAX_ITER)
                    row_ptr[x + i] = Vec3b(0, 0, 0);
                } else {
                    // Same coloring logic as before
                    unsigned char color_val = (unsigned char)(255.0 * iter / 10);
                    row_ptr[x + i] = Vec3b(color_val, color_val, 0);
                }
            }
        }

        // --- Scalar Cleanup Loop (for width % 8 != 0) ---
        for (; x < WIDTH; ++x) {
            // (Copy-paste logic from serial version for single pixel)
            double real = REAL_MIN + (double)x / WIDTH * (REAL_MAX - REAL_MIN);
            double imag = IMAG_MIN + (double)y / HEIGHT * (IMAG_MAX - IMAG_MIN);
            complex<double> z(real, imag);
            int iter = 0;
            while (iter < MAX_ITER) {
                if (std::norm(z) > ESCAPE_RADIUS_SQ) break;
                z = std::pow(z, POWER_N) + C;
                iter++;
            }
            if (iter == MAX_ITER) {
                row_ptr[x] = Vec3b(0, 0, 0);
            } else {
                unsigned char color_val = (unsigned char)(255.0 * iter / 10);
                row_ptr[x] = Vec3b(color_val, color_val, 0);
            }
        }
    }
}

int main() {
    // 1. Setup Images
    Mat image_serial(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));
    Mat image_parallel(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));

    cout << "--- Multi-Julia Set Configuration ---" << endl;
    cout << "Image: " << WIDTH << "x" << HEIGHT << endl;
    cout << "Max Iterations: " << MAX_ITER << endl;
    cout << "Power N: " << POWER_N << endl;
    cout << "Max Threads: " << omp_get_max_threads() << endl;
    cout << "-----------------------------------" << endl;

    // 2. Run Serial (with RDTSC timing)
    cout << "Running Serial..." << endl;
    unsigned long long start_serial = __rdtsc();
    
    generate_julia_serial(image_serial);
    
    unsigned long long end_serial = __rdtsc();
    unsigned long long time_serial = end_serial - start_serial;
    cout << "Serial Time:   " << time_serial << " clocks." << endl;

    // 3. Run Parallel (OpenMP) (with RDTSC timing)
    cout << "Running Parallel (OpenMP)..." << endl;
    unsigned long long start_parallel = __rdtsc();
    
    generate_julia_parallel(image_parallel);
    
    unsigned long long end_parallel = __rdtsc();
    unsigned long long time_parallel = end_parallel - start_parallel;
    cout << "Parallel Time: " << time_parallel << " clocks." << endl;

    // 4. Run Parallel (OpenMP + AVX)
    Mat image_avx(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));
    cout << "Running Parallel (OpenMP + AVX)..." << endl;
    unsigned long long start_avx = __rdtsc(); // Or omp_get_wtime()
    double start_avx_time = omp_get_wtime();
    
    generate_julia_avx(image_avx);
    
    double end_avx_time = omp_get_wtime();
    unsigned long long end_avx = __rdtsc();
    unsigned long long time_avx = end_avx - start_avx;
    cout << "AVX Time: " << (end_avx_time - start_avx_time) << " seconds." << endl;

    // 4. Results & Speedup
    cout << "-----------------------------------" << endl;
    cout << fixed << setprecision(2);
    cout << "Speedup: " << (double)time_serial / time_parallel << "x" << endl;
    cout << "Speedup avx (vs Serial): " << (double)time_serial / (end_avx - start_avx) << "x" << endl;

    // 5. Save Images
    imwrite("output/julia_serial.jpg", image_serial);
    imwrite("output/julia_parallel.jpg", image_parallel);
    imwrite("output/julia_avx.jpg", image_avx);
    cout << "Images saved." << endl;

    return 0;
}