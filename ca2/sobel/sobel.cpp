#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <x86intrin.h> // For __rdtsc()
#include <opencv2/opencv.hpp> // OpenCV Library

using namespace std;
using namespace cv;

// --- Filter Kernels (float) ---
// Gaussian Kernel
const float gaussian_kernel[9] = {
    1.0f, 2.0f, 1.0f,
    2.0f, 4.0f, 2.0f,
    1.0f, 2.0f, 1.0f
};
// Normalization factor (1+2+1+2+4+2+1+2+1)
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
 * @brief Simple serial 3x3 convolution function (Used for Gaussian)
 * @param src Input Mat (must be CV_32F - float)
 * @param dst Output Mat (same size as src, CV_32F)
 * @param kernel 3x3 kernel (9-element array)
 * @param norm_factor Normalization factor (to divide by at the end)
 */
void convolve_serial(const Mat& src, Mat& dst, const float* kernel, float norm_factor) {
    int rows = src.rows;
    int cols = src.cols;

    // Ensure output Mat is correctly sized and typed
    dst.create(rows, cols, CV_32F);
    
    // Loop from 1 to rows-1 (skipping borders for simplicity)
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

int main() {
    // --- Step 0: Load and Prepare Image ---
    Mat base_img = imread("../assets/assets/base.jpg");
    if (base_img.empty()) {
        cout << "Error: base.jpg image not found. Make sure it's in the same directory." << endl;
        return -1;
    }

    Mat gray_img;
    cvtColor(base_img, gray_img, COLOR_BGR2GRAY);

    Mat gray_float;
    gray_img.convertTo(gray_float, CV_32F); // Convert to float for calculations

    // Prepare output matrices
    Mat blurred(gray_float.size(), CV_32F);
    
    // [EDIT] No longer need edge_x and edge_y
    // Mat edge_x(gray_float.size(), CV_32F);
    // Mat edge_y(gray_float.size(), CV_32F);
    
    Mat magnitude_serial(gray_float.size(), CV_32F);
    Mat magnitude_serial_8u(gray_float.size(), CV_8U); // Final 8-bit output

    cout << "--- Running *Optimized* Serial Sobel Pipeline ---" << endl;
    cout << "Image Size: " << gray_img.cols << "x" << gray_img.rows << endl;
    
    // --- Timing ---
    unsigned long long start_time, end_time;
    start_time = __rdtsc();

    // --- Step 1: Gaussian Blur (Still separate) ---
    convolve_serial(gray_float, blurred, gaussian_kernel, GAUSSIAN_NORM);
    
    // [EDIT] Steps 2, 3, and 4 are now FUSED into a single loop

    int rows = gray_float.rows;
    int cols = gray_float.cols;

    // Loop from 1 to rows-1 (skipping borders)
    for (int y = 1; y < rows - 1; ++y) {
        // Get pointers to blurred image rows
        const float* p_top = blurred.ptr<float>(y - 1);
        const float* p_mid = blurred.ptr<float>(y);
        const float* p_bot = blurred.ptr<float>(y + 1);
        
        // Get pointer to final magnitude output row
        float* p_mag = magnitude_serial.ptr<float>(y);

        for (int x = 1; x < cols - 1; ++x) {
            
            // --- Calculate Sobel X (Step 2) ---
            // We can hard-code kernel lookups since 0-weighted cells are skipped
            float gx = (p_top[x - 1] * sobel_x_kernel[0]) + (p_top[x + 1] * sobel_x_kernel[2]) +
                       (p_mid[x - 1] * sobel_x_kernel[3]) + (p_mid[x + 1] * sobel_x_kernel[5]) +
                       (p_bot[x - 1] * sobel_x_kernel[6]) + (p_bot[x + 1] * sobel_x_kernel[8]);
            
            // --- Calculate Sobel Y (Step 3) ---
            // We can hard-code kernel lookups since 0-weighted cells are skipped
            float gy = (p_top[x - 1] * sobel_y_kernel[0]) + (p_top[x] * sobel_y_kernel[1]) + (p_top[x + 1] * sobel_y_kernel[2]) +
                       (p_bot[x - 1] * sobel_y_kernel[6]) + (p_bot[x] * sobel_y_kernel[7]) + (p_bot[x + 1] * sobel_y_kernel[8]);
            
            // --- Calculate Magnitude (Step 4) ---
            p_mag[x] = sqrt(gx * gx + gy * gy);
        }
    }


    // --- End Timing ---
    end_time = __rdtsc();
    cout << "Optimized Serial Time (clocks): " << (end_time - start_time) << endl;

    // Convert float output to 8-bit (unsigned char) for saving
    // cv::saturate_cast handles clamping to [0, 255]
    magnitude_serial.convertTo(magnitude_serial_8u, CV_8U);

    // Save result
    imwrite("sobel_serial_result.jpg", magnitude_serial_8u);
    cout << "Serial result saved to sobel_serial_result.jpg" << endl;

    return 0;
}

