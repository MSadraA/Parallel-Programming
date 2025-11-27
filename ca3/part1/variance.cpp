#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include "ipp.h"       
#include <immintrin.h> 
#include <pmmintrin.h> 
#include "omp.h"

using namespace std;

const size_t ARRAY_SIZE = 16000000;
const size_t ALIGNMENT = 16; // 16 bytes equivalent to 128 bits for sse instructions
const size_t limit = (ARRAY_SIZE / 4) * 4;

float serial_optimized_calculation(float* data){

    float sum_parts[4]; // Using float to match SIMD precision
    float total_sum = 0.0f;
    float mean = 0.0f;
    float total_sq_diff = 0.0f;
    size_t i = 0; // Use size_t for index

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
    mean = total_sum / ARRAY_SIZE;

    // --- Serial Pass 2: Calculate Sum of Squared Differences ---
    sum_parts[0] = 0.0f; sum_parts[1] = 0.0f; sum_parts[2] = 0.0f; sum_parts[3] = 0.0f;
    for(i = 0 ; i < limit ; i+=4){
        float diff0 = data[i] - mean;
        float diff1 = data[i+1] - mean;
        float diff2 = data[i+2] - mean;
        float diff3 = data[i+3] - mean;
        sum_parts[0] += diff0 * diff0;
        sum_parts[1] += diff1 * diff1;
        sum_parts[2] += diff2 * diff2;
        sum_parts[3] += diff3 * diff3;
    }
    total_sq_diff = sum_parts[0] + sum_parts[1] + sum_parts[2] + sum_parts[3];

    // Start remainder loop from 'limit'
    for (i = limit; i < ARRAY_SIZE; ++i) {
        float diff = data[i] - mean;
        total_sq_diff += diff * diff;
    }
    return total_sq_diff / ARRAY_SIZE;
}

float simd_calculation(float* data){
    // --- Parallel variables ---
    __m128 sums;
    __m128 sum_vec;
    __m128 mean_vec;
    float total_sum = 0.0f;
    float total_sq_diff = 0.0f;
    float mean = 0.0f;
    size_t i = 0;

    // --- SIMD Pass 1: Calculate Mean ---
    sum_vec = _mm_setzero_ps(); // make a pack of 4 single-precision values by setting each element to 0
    
    for (i = 0; i < limit; i += 4) {
        __m128 data_vec = _mm_load_ps(data + i); // load 4 single-precision values from ALIGNED memory
        sum_vec = _mm_add_ps(sum_vec, data_vec); // add 4 single-precision values
    }
    sums = _mm_hadd_ps(sum_vec, sum_vec); // add pairs of single-precision values
    sums = _mm_hadd_ps(sums, sums); 
    total_sum = _mm_cvtss_f32(sums);
    
    for (i = limit; i < ARRAY_SIZE; ++i) { total_sum += data[i]; } // add remaining elements
    mean = static_cast<float>(total_sum / ARRAY_SIZE);

    // --- SIMD Pass 2: Calculate Sum of Squared Differences ---
    mean_vec = _mm_set1_ps(mean); // Use the mean calculated by SIMD
    sum_vec = _mm_setzero_ps(); 
    
    for(i = 0 ; i < limit; i += 4){
        __m128 data_vec = _mm_load_ps(data + i); // Use ALIGNED load
        __m128 diff_vec = _mm_sub_ps(data_vec, mean_vec);
        __m128 sq_diff_vec = _mm_mul_ps(diff_vec, diff_vec);
        sum_vec = _mm_add_ps(sum_vec, sq_diff_vec);
    }
    sums = _mm_hadd_ps(sum_vec, sum_vec);
    sums = _mm_hadd_ps(sums, sums);
    total_sq_diff = _mm_cvtss_f32(sums);

    for (i = limit; i < ARRAY_SIZE; ++i) { // add remaining elements
        float diff = data[i] - mean;
        total_sq_diff += diff * diff;
    }
    return static_cast<float>(total_sq_diff / ARRAY_SIZE);
}

float openmp_calculation(float* data) {
    float total_sum = 0.0f;
    float total_sq_diff = 0.0f;
    float mean = 0.0f;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        size_t raw_chunk = ARRAY_SIZE / nthreads;
        size_t chunk_aligned = (raw_chunk / 4) * 4; // each chunk must be a multiple of 4
        
        size_t start = tid * chunk_aligned;
        size_t end = (tid == nthreads - 1) ? ARRAY_SIZE : start + chunk_aligned;

        float partial_sum = 0.0f;
        float partial_sq = 0.0f;
        size_t i;

        size_t limit = start + ((end - start) / 4) * 4;

        for (i = start; i < limit; i += 4) {
            partial_sum += data[i];
            partial_sum += data[i+1];
            partial_sum += data[i+2];
            partial_sum += data[i+3];
        }

        for (; i < end; ++i) {
            partial_sum += data[i];
        }

        #pragma omp atomic
        total_sum += partial_sum;

        #pragma omp barrier 

        #pragma omp master
        mean = total_sum / ARRAY_SIZE;

        #pragma omp barrier
        
        for (i = start; i < limit; i += 4) {
            float d0 = data[i] - mean;
            float d1 = data[i+1] - mean;
            float d2 = data[i+2] - mean;
            float d3 = data[i+3] - mean;
            
            partial_sq += d0 * d0;
            partial_sq += d1 * d1;
            partial_sq += d2 * d2;
            partial_sq += d3 * d3;
        }

        for (; i < end; ++i) {
            float d = data[i] - mean;
            partial_sq += d * d;
        }

        #pragma omp atomic
        total_sq_diff += partial_sq;
    }

    return total_sq_diff / ARRAY_SIZE;
}

float openmp_simd_calculation(float* data) {
    float total_sum = 0.0f;
    float total_sq_diff = 0.0f;
    float mean = 0.0f;

    #pragma omp parallel
    {
        __m128 sum_vec;
        __m128 mean_vec;

        float partial_sum = 0.0f;
        float partial_sq = 0.0f;

        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        size_t raw_chunk = ARRAY_SIZE / nthreads;
        size_t chunk_aligned = (raw_chunk / 4) * 4; 
        
        size_t start = tid * chunk_aligned;
        size_t end = (tid == nthreads - 1) ? ARRAY_SIZE : start + chunk_aligned;
        
        size_t limit = start + ((end - start) / 4) * 4;
        size_t i;

        sum_vec = _mm_setzero_ps();
        for (i = start; i < limit; i += 4) {
            __m128 data_vec = _mm_load_ps(data + i); 
            sum_vec = _mm_add_ps(sum_vec, data_vec);
        }

        __m128 sums = _mm_hadd_ps(sum_vec, sum_vec);
        sums = _mm_hadd_ps(sums, sums); 
        partial_sum = _mm_cvtss_f32(sums);

        for (; i < end; ++i) { 
            partial_sum += data[i]; 
        } 

        #pragma omp atomic
        total_sum += partial_sum;

        #pragma omp barrier 

        #pragma omp master
        mean = total_sum / ARRAY_SIZE;
        
        #pragma omp barrier
        
        mean_vec = _mm_set1_ps(mean);
        sum_vec = _mm_setzero_ps(); 
    
        for(i = start; i < limit; i += 4){
            __m128 data_vec = _mm_load_ps(data + i);
            __m128 diff_vec = _mm_sub_ps(data_vec, mean_vec);
            __m128 sq_diff_vec = _mm_mul_ps(diff_vec, diff_vec);
            sum_vec = _mm_add_ps(sum_vec, sq_diff_vec);
        }

        sums = _mm_hadd_ps(sum_vec, sum_vec);
        sums = _mm_hadd_ps(sums, sums);
        partial_sq = _mm_cvtss_f32(sums);

        for (; i < end; ++i) { 
            float diff = data[i] - mean;
            partial_sq += diff * diff;
        }

        #pragma omp atomic
        total_sq_diff += partial_sq;
    }

    return total_sq_diff / ARRAY_SIZE;
}

int main()
{
    Ipp64u start, end;
    Ipp64u time1, time2, time3 , time4;
    
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

    start = ippGetCpuClocks();
    float serial_variance = serial_optimized_calculation(data);
    end = ippGetCpuClocks();
    time1 = end - start;

    start = ippGetCpuClocks();
    float simd_variance = simd_calculation(data);
    end = ippGetCpuClocks();
    time2 = end - start;

    start = ippGetCpuClocks();
    float openmp_variance = openmp_calculation(data);
    end = ippGetCpuClocks();
    time3 = end - start;

    start = ippGetCpuClocks();
    float openmp_simd_variance = openmp_simd_calculation(data);
    end = ippGetCpuClocks();
    time4 = end - start;

    // --- Final Results ---
    cout << "\nSerial Variance: " << serial_variance << " clock cycles: " << (long long)time1 << endl;
    cout << "SIMD Variance: " << simd_variance << " clock cycles: " << (long long)time2 << endl;
    cout << "OpenMP Variance: " << openmp_variance << " clock cycles: " << (long long)time3 << endl;
    cout << "OpenMP SIMD Variance: " << openmp_simd_variance << " clock cycles: " << (long long)time4 << endl;
    
    // Cast to double *before* division to avoid integer division
    cout << "Speedup SIMD: " << std::setprecision(4) << ((double)time1 / (double)time2) << "x" << endl;
    cout << "Speedup OpenMP: " << std::setprecision(4) << ((double)time1 / (double)time3) << "x" << endl;
    cout << "Speedup OpenMP + SIMD: " << std::setprecision(4) << ((double)time1 / (double)time4) << "x" << endl;

    _mm_free(data); 
}