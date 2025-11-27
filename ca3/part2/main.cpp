#include <iostream>
#include <complex>      // Standard complex number class
#include <opencv2/opencv.hpp>
#include <omp.h>        // Included for timing (omp_get_wtime)

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

int main() {
    // Create a black image (CV_8UC3 means 8-bit Unsigned, 3 Channels/Colors)
    Mat image(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));

    cout << "Generating Julia Set (Serial)..." << endl;
    cout << "Image Size: " << WIDTH << "x" << HEIGHT << endl;
    cout << "Max Iterations: " << MAX_ITER << endl;

    // Start Timing
    double start_time = omp_get_wtime();

    // Run the serial algorithm
    generate_julia_serial(image);

    // End Timing
    double end_time = omp_get_wtime();

    cout << "Done." << endl;
    cout << "Serial Execution Time: " << (end_time - start_time) << " seconds." << endl;

    // Save the result
    imwrite("julia_serial.jpg", image);
    cout << "Image saved to julia_serial.jpg" << endl;

    return 0;
}