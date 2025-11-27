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
const double ESCAPE_RADIUS_SQ = 2.0;

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

        // Loop over every column (x)
        for (int x = 0; x < WIDTH; ++x) {
            
            // --- Step 1: Map pixel coordinate (x, y) to complex plane (real, imag) ---
            double real = REAL_MIN + (double)x / WIDTH * (REAL_MAX - REAL_MIN);
            double imag = IMAG_MIN + (double)y / HEIGHT * (IMAG_MAX - IMAG_MIN);

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
void generate_julia_parallel(Mat& image) {
    
    // Loop variables declared outside to be compatible with OpenMP directive syntax
    int y;
    
    // --- OpenMP Directive Explanation ---
    // parallel for: Automatically splits the loop iterations among threads.
    // schedule(dynamic): Assigns chunks of work dynamically. Crucial here because
    //                    some rows (outside the set) finish fast, while others (inside) take long.
    // shared(image): The image matrix is shared (threads write to different parts).
    #pragma omp parallel for schedule(dynamic) shared(image) private(y)
    for (y = 0; y < HEIGHT; ++y) {
        
        // Get pointer to the current row.
        // Since 'image' is shared, multiple threads access it, but they access
        // different rows 'y', so no race condition occurs here.
        Vec3b* row_ptr = image.ptr<Vec3b>(y);

        // Inner loop (x) runs serially within each thread for the assigned row 'y'
        for (int x = 0; x < WIDTH; ++x) {
            
            // --- Step 1: Map pixel to complex plane ---
            // Optimization: Pre-calculate constants to avoid division in loop if possible,
            // but for readability, we keep the formula.
            double real = REAL_MIN + (double)x / WIDTH * (REAL_MAX - REAL_MIN);
            double imag = IMAG_MIN + (double)y / HEIGHT * (IMAG_MAX - IMAG_MIN);

            complex<double> z(real, imag);
            
            // --- Step 2: Iterate ---
            int iter = 0;
            while (iter < MAX_ITER) {
                // Check squared norm > 4.0
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


int main() {
    // Create a black image (CV_8UC3 means 8-bit Unsigned, 3 Channels/Colors)
    Mat image_serial(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));
    Mat image_parallel(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));

    cout << "--- Multi-Julia Set Configuration ---" << endl;
    cout << "Image: " << WIDTH << "x" << HEIGHT << endl;
    cout << "Max Iterations: " << MAX_ITER << endl;
    cout << "Power N: " << POWER_N << endl;
    cout << "Max Threads: " << omp_get_max_threads() << endl; // Check available cores
    cout << "-----------------------------------" << endl;

    // Start Timing
    // unsigned long long start_clock, end_clock;

    // 2. Run Serial
    cout << "Running Serial..." << endl;
    double start_serial = omp_get_wtime();
    generate_julia_serial(image_serial);
    double end_serial = omp_get_wtime();
    double time_serial = end_serial - start_serial;
    cout << "Serial Time:   " << time_serial << " seconds." << endl;

    // 3. Run Parallel (OpenMP)
    cout << "Running Parallel (OpenMP)..." << endl;
    double start_parallel = omp_get_wtime();
    generate_julia_parallel(image_parallel);
    double end_parallel = omp_get_wtime();
    double time_parallel = end_parallel - start_parallel;
    cout << "Parallel Time: " << time_parallel << " seconds." << endl;

// 4. Results & Speedup
    cout << "-----------------------------------" << endl;
    cout << "Speedup: " << time_serial / time_parallel << "x" << endl;

    // 5. Save Images
    imwrite("output/julia_serial.jpg", image_serial);
    imwrite("output/julia_parallel.jpg", image_parallel);
    cout << "Images saved." << endl;

    return 0;
}