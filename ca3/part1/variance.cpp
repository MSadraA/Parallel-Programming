#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include "ipp.h"       
#include <immintrin.h> 
#include <pmmintrin.h> 

using namespace std;

const size_t ARRAY_SIZE = 16000000;
const size_t ALIGNMENT = 16; // 16 bytes equivalent to 128 bits for sse instructions

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

    //initialize array (using Aligned memory allocation)
    float* data = static_cast<float*>(_mm_malloc(ARRAY_SIZE * sizeof(float), ALIGNMENT)); // Pointer to multiple of 16 bytes
    if (!data) {
        cout << "Error: Memory allocation failed." << endl;
        return 1;
    }
    srand(static_cast<unsigned int>(time(NULL)));
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    cout << "--- Variance Calculation ---" << endl;
    cout << "Array Size: " << ARRAY_SIZE << " floats" << endl;
    cout << std::fixed << std::setprecision(10); 

    const size_t limit = (ARRAY_SIZE / 4) * 4;

    // --- Optimized Serial (Using 1 loop per pass, unrolled) ---
    start = ippGetCpuClocks();

    // --- Serial Pass 1: Calculate Mean ---
    sum_parts[0] = 0.0f; sum_parts[1] = 0.0f; sum_parts[2] = 0.0f; sum_parts[3] = 0.0f;
    for (i = 0; i < limit ; i+=4){
        sum_parts[0] += data[i];
        sum_parts[1] += data[i+1];
        sum_parts[2] += data[i+2];
        sum_parts[3] += data[i+3];
    }
    total_sum = sum_parts[0] + sum_parts[1] + sum_parts[2] + sum_parts[3];

    for (i = limit; i < ARRAY_SIZE; ++i) { total_sum += data[i]; } // add remaining elements
    serial_mean = total_sum / ARRAY_SIZE;

    // --- Serial Pass 2: Calculate Sum of Squared Differences ---
    sum_parts[0] = 0.0f; sum_parts[1] = 0.0f; sum_parts[2] = 0.0f; sum_parts[3] = 0.0f;
    for(i = 0 ; i < limit ; i+=4){
        float diff0 = data[i] - serial_mean;
        float diff1 = data[i+1] - serial_mean;
        float diff2 = data[i+2] - serial_mean;
        float diff3 = data[i+3] - serial_mean;
        sum_parts[0] += diff0 * diff0;
        sum_parts[1] += diff1 * diff1;
        sum_parts[2] += diff2 * diff2;
        sum_parts[3] += diff3 * diff3;
    }
    total_sq_diff = sum_parts[0] + sum_parts[1] + sum_parts[2] + sum_parts[3];

    // Start remainder loop from 'limit'
    for (i = limit; i < ARRAY_SIZE; ++i) {
        float diff = data[i] - serial_mean;
        total_sq_diff += diff * diff;
    }
    serial_variance = total_sq_diff / ARRAY_SIZE;
    
    end = ippGetCpuClocks();
    time1 = end - start;

    // --- Parallel SIMD (Using _mm_load_ps with aligned memory) ---
    start = ippGetCpuClocks();

    // --- SIMD Pass 1: Calculate Mean ---
    sum_vec = _mm_setzero_ps(); // make a pack of 4 single-precision values by setting each element to 0
    
    for (i = 0; i < limit; i += 4) {
        __m128 data_vec = _mm_load_ps(data + i); // load 4 single-precision values from ALIGNED memory
        sum_vec = _mm_add_ps(sum_vec, data_vec); // add 4 single-precision values
    }
    sums = _mm_hadd_ps(sum_vec, sum_vec); // add pairs of single-precision values
    sums = _mm_hadd_ps(sums, sums); 
    total_sum_simd = _mm_cvtss_f32(sums);
    
    for (i = limit; i < ARRAY_SIZE; ++i) { total_sum_simd += data[i]; } // add remaining elements
    mean_simd = static_cast<float>(total_sum_simd / ARRAY_SIZE);

    // --- SIMD Pass 2: Calculate Sum of Squared Differences ---
    mean_vec = _mm_set1_ps(mean_simd); // Use the mean calculated by SIMD
    sum_vec = _mm_setzero_ps(); 
    
    for(i = 0 ; i < limit; i += 4){
        __m128 data_vec = _mm_load_ps(data + i); // Use ALIGNED load
        __m128 diff_vec = _mm_sub_ps(data_vec, mean_vec);
        __m128 sq_diff_vec = _mm_mul_ps(diff_vec, diff_vec);
        sum_vec = _mm_add_ps(sum_vec, sq_diff_vec);
    }
    sums = _mm_hadd_ps(sum_vec, sum_vec);
    sums = _mm_hadd_ps(sums, sums);
    total_sq_diff_simd = _mm_cvtss_f32(sums);

    for (i = limit; i < ARRAY_SIZE; ++i) { // add remaining elements
        float diff = data[i] - mean_simd;
        total_sq_diff_simd += diff * diff;
    }
    parallel_variance = static_cast<float>(total_sq_diff_simd / ARRAY_SIZE);
    
    end = ippGetCpuClocks();
    time2 = end - start;

    // --- Final Results ---
    cout << "\nSerial Variance: " << serial_variance << " clock cycles: " << (long long)time1 << endl;
    cout << "Parallel Variance: " << parallel_variance << " clock cycles: " << (long long)time2 << endl;
    
    // Cast to double *before* division to avoid integer division
    cout << "Speedup: " << std::setprecision(4) << ((double)time1 / (double)time2) << "x" << endl;

    _mm_free(data); 
}