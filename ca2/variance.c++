#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include "ipp.h"       // For ippGetCpuClocks
#include <immintrin.h> // Main header for all SIMD intrinsics
#include <pmmintrin.h> // For _mm_hadd_ps (SSE3)

using namespace std;

const size_t ARRAY_SIZE = 16000000;

float horizontal_sum_sse(__m128 v) {
    __m128 sums;
    sums = _mm_hadd_ps(v, v); // add pairs of single-precision values like [(3+2), (1+0), (3+2), (1+0)]
    sums = _mm_hadd_ps(sums, sums); // Final horizontal add: [(3+2+1+0), ...]
    return _mm_cvtss_f32(sums); // convert scalar single-precision to float 32-bit value
}

int main()
{
    Ipp64u start, end;
    Ipp64u time1, time2;
    
    // --- Serial variables ---
    float sum_parts[4]; // Using float to match SIMD precision
    float total_sum = 0.0f;
    float serial_mean = 0.0f;
    float total_sq_diff = 0.0f;
    float serial_variance = 0.0f;
    size_t i = 0; // Use size_t for index

    // --- Parallel variables ---
    __m128 sums;
    __m128 sum_vec;
    __m128 mean_vec;
    float parallel_variance = 0.0f;
    float total_sum_simd = 0.0f;
    float total_sq_diff_simd = 0.0f;
    float mean_simd = 0.0f;

    //initialize array (using C-style new[] as per the template)
    float* data = new float [ARRAY_SIZE]; 
    if (!data) {
        cout << "Error: Memory allocation failed." << endl;
        return 1;
    }
    srand(static_cast<unsigned int>(time(NULL)));
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    cout << "--- Variance Calculation (C-Style Logic, C++ Names) ---" << endl;
    cout << "Array Size: " << ARRAY_SIZE << " floats" << endl;
    cout << std::fixed << std::setprecision(10); 

    // ==========================================================
    // --- Optimized Serial (Using 4 separate loops from template) ---
    // ==========================================================
    start = ippGetCpuClocks();

    // Serial Pass 1: Calculate Mean
    sum_parts[0] = 0.0f; sum_parts[1] = 0.0f; sum_parts[2] = 0.0f; sum_parts[3] = 0.0f; 
    for (i = 0; i + 3 < ARRAY_SIZE; i += 4)
        sum_parts[0] += data[i];
    for (i = 0; i + 3 < ARRAY_SIZE; i += 4)
        sum_parts[1] += data[i + 1];
    for (i = 0; i + 3 < ARRAY_SIZE; i += 4)
        sum_parts[2] += data[i + 2];
    for (i = 0; i + 3 < ARRAY_SIZE; i += 4)
        sum_parts[3] += data[i + 3];
    // (Note: Template does not show handling for non-multiples of 4)
    
    total_sum = sum_parts[0] + sum_parts[1] + sum_parts[2] + sum_parts[3];
    serial_mean = total_sum / ARRAY_SIZE;

    // Serial Pass 2: Calculate Sum of Squared Differences
    sum_parts[0] = 0.0f; sum_parts[1] = 0.0f; sum_parts[2] = 0.0f; sum_parts[3] = 0.0f;
    i = 0; 
    for (i = 0; i + 3 < ARRAY_SIZE; i += 4) {
        float diff = data[i] - serial_mean;
        sum_parts[0] += diff * diff;
    }
    for (i = 0; i + 3 < ARRAY_SIZE; i += 4) {
        float diff = data[i+1] - serial_mean;
        sum_parts[1] += diff * diff;
    }
    for (i = 0; i + 3 < ARRAY_SIZE; i += 4) {
        float diff = data[i+2] - serial_mean;
        sum_parts[2] += diff * diff;
    }
    for (i = 0; i + 3 < ARRAY_SIZE; i += 4) {
        float diff = data[i+3] - serial_mean;
        sum_parts[3] += diff * diff;
    }

    total_sq_diff = sum_parts[0] + sum_parts[1] + sum_parts[2] + sum_parts[3];
    serial_variance = total_sq_diff / ARRAY_SIZE;
    
    end = ippGetCpuClocks();
    time1 = end - start;

    // ==========================================================
    // --- Parallel SIMD (Using _mm_loadu_ps to match new[]) ---
    // ==========================================================
    start = ippGetCpuClocks();

    // --- SIMD Pass 1: Calculate Mean ---
    sum_vec = _mm_setzero_ps(); // make a pack of 4 single-precision values by setting each element to 0
    i = 0; 
    for (; i + 3 < ARRAY_SIZE; i += 4) {
        __m128 data_vec = _mm_loadu_ps(data + i); // load 4 single-precision values from UNALIGNED memory
        sum_vec = _mm_add_ps(sum_vec, data_vec); // add 4 single-precision values
    }
    sums = _mm_hadd_ps(sum_vec, sum_vec); // add pairs of single-precision values
    sums = _mm_hadd_ps(sums, sums); 
    total_sum_simd = (double)_mm_cvtss_f32(sums); // (Cast to double for precision)
    
    for (; i < ARRAY_SIZE; ++i) { total_sum_simd += data[i]; } // add remaining elements
    mean_simd = static_cast<float>(total_sum_simd / ARRAY_SIZE);

    // --- SIMD Pass 2: Calculate Sum of Squared Differences ---
    mean_vec = _mm_set1_ps(mean_simd); // Use the mean calculated by SIMD
    sum_vec = _mm_setzero_ps(); 
    i = 0; 
    
    for(i = 0 ; i + 3 < ARRAY_SIZE; i += 4){
        __m128 data_vec = _mm_loadu_ps(data + i); // Use UNALIGNED load
        __m128 diff_vec = _mm_sub_ps(data_vec, mean_vec);
        __m128 sq_diff_vec = _mm_mul_ps(diff_vec, diff_vec);
        sum_vec = _mm_add_ps(sum_vec, sq_diff_vec);
    }
    sums = _mm_hadd_ps(sum_vec, sum_vec);
    sums = _mm_hadd_ps(sums, sums);
    total_sq_diff_simd = (double)_mm_cvtss_f32(sums); // (Cast to double for precision)

    for (; i < ARRAY_SIZE; ++i) { // add remaining elements
        float diff = data[i] - mean_simd;
        total_sq_diff_simd += diff * diff;
    }
    parallel_variance = static_cast<float>(total_sq_diff_simd / ARRAY_SIZE);
    
    end = ippGetCpuClocks();
    time2 = end - start;

    // ==========================================================
    // --- Final Results ---
    // ==========================================================
    cout << "\nSerial Variance: " << serial_variance << " clock cycles: " << (long long)time1 << endl;
    cout << "Parallel Variance: " << parallel_variance << " clock cycles: " << (long long)time2 << endl;
    
    // Cast to double *before* division to avoid integer division
    cout << "Speedup: " << std::setprecision(4) << ((double)time1 / (double)time2) << "x" << endl;

    delete[] data; // Free the memory allocated with new[]
}