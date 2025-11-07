#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>      // For std::setprecision
#include <x86intrin.h>  // For __rdtsc()
#include <malloc.h>     // For _mm_malloc / _mm_free (Aligned allocation)
#include <cstdio>       // For fopen, fread, fclose
#include <string>       // For std::string
#include <cstring>      // For memcmp (verification)

using namespace std;

// --- Network Structure Definitions ---
const int INPUT_SIZE = 8;
const int HIDDEN_SIZE = 16;
const int OUTPUT_SIZE = 8;

const size_t ALIGNMENT = 32; 

// === Pre-defined Functions ===

inline float activation_relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

inline float horizontal_sum_avx(__m256 v) {
    // 1. Add upper 128 bits to lower 128 bits
    // v_high = {f4, f5, f6, f7}
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    // v_low  = {f0, f1, f2, f3}
    __m128 v_low  = _mm256_castps256_ps128(v);
    // v_sum128 = {f0+f4, f1+f5, f2+f6, f3+f7}
    __m128 v_sum128 = _mm_add_ps(v_low, v_high);
    
    __m128 v_hadd1 = _mm_hadd_ps(v_sum128, v_sum128);
    __m128 v_hadd2 = _mm_hadd_ps(v_hadd1, v_hadd1);
    
    // 3. Extract the final scalar sum
    return _mm_cvtss_f32(v_hadd2);
}

void forward_layer_parallel(
    float* output_vector,
    const float* input_vector,
    const float* weights_transposed,
    const float* bias_vector,
    int num_inputs,
    int num_outputs
) {
    // Calculate the number of 8-float chunks
    const int num_chunks_8 = num_inputs / 8;
    
    // Loop over each output neuron (this loop remains serial)
    for (int j = 0; j < num_outputs; ++j) {
        
        // Pointer to the start of the current weight row
        const float* p_weight_row = weights_transposed + (j * num_inputs);
        
        // v_sum will accumulate dot products in 8 parallel "lanes"
        __m256 v_sum = _mm256_setzero_ps();
        
        // --- Vectorized Inner Loop ---
        // Process inputs 8 at a time
        for (int i = 0; i < num_chunks_8; ++i) {
            // Load 8 inputs
            __m256 v_input = _mm256_load_ps(input_vector + i * 8);
            // Load 8 weights
            __m256 v_weight = _mm256_load_ps(p_weight_row + i * 8);
            
            // Fused Multiply-Add: v_sum = v_sum + (v_input * v_weight)
            v_sum = _mm256_fmadd_ps(v_input, v_weight, v_sum);
        }
        
        // --- Horizontal Sum ---
        // Sum the 8 partial sums from v_sum into a single scalar
        float sum = horizontal_sum_avx(v_sum);
        
        // // --- Remainder Loop (Scalar) ---
        // // Handle any inputs that weren't a multiple of 8
        // for (int i = num_chunks_8 * 8; i < num_inputs; ++i) {
        //     sum += input_vector[i] * p_weight_row[i];
        // }

        // Add bias and apply activation
        output_vector[j] = activation_relu(sum + bias_vector[j]);
    }
}

// void forward_layer_parallel(
//     float* output_vector,
//     const float* input_vector,
//     const float* weights_transposed,
//     const float* bias_vector,
//     int num_inputs,
//     int num_outputs
// ) {
//     for (int j = 0; j < num_outputs; ++j) {
//         __m256 sum_vec = _mm256_setzero_ps();
//         int i = 0;
//         for (; i <= num_inputs - 8; i += 8) {
//             __m256 input_vec = _mm256_load_ps(input_vector + i);
//             __m256 weight_vec = _mm256_load_ps(weights_transposed + j * num_inputs + i);
//             sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
//         }

//         // Horizontal add to get the sum of all elements in sum_vec
//         float sum_array[8];
//         _mm256_store_ps(sum_array, sum_vec);
//         float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
//                     sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

//         // Handle remaining elements
//         // for (; i < num_inputs; ++i) {
//         //     sum += input_vector[i] * weights_transposed[j * num_inputs + i];
//         // }

//         // Add bias and apply activation function
//         output_vector[j] = activation_relu(sum + bias_vector[j]);
//     }
// }

void forward_layer_serial(
    float* output_vector,
    const float* input_vector,
    const float* weights_transposed,
    const float* bias_vector,
    int num_inputs,
    int num_outputs
) {
    // Loop over each output neuron
    for (int j = 0; j < num_outputs; ++j) {
        
        // Calculate the weighted sum (Dot Product)
        float sum = 0.0f;
        
        // This inner loop is cache-friendly because:
        // 1. input_vector is small and stays in L1 cache.
        // 2. weights_transposed[...] is read contiguously from memory.
        const float* p_weight_row = weights_transposed + (j * num_inputs);
        
        for (int i = 0; i < num_inputs; ++i) {
            sum += input_vector[i] * p_weight_row[i];
        }
        
        // Add bias and apply activation function
        output_vector[j] = activation_relu(sum + bias_vector[j]);
    }
}

// === New Helper Functions ===

/**
 * @brief Helper function to read binary data from a file.
 */
bool read_from_file(const string& filename, float* data, size_t num_elements) {
    FILE* f = fopen(filename.c_str(), "rb"); // "rb" = read binary
    if (f == NULL) {
        cerr << "Error: Could not open file " << filename << endl;
        return false;
    }
    
    size_t read_count = fread(data, sizeof(float), num_elements, f);
    fclose(f);
    
    if (read_count != num_elements) {
        cerr << "Error: Read incorrect number of elements from " << filename << endl;
        return false;
    }
    return true;
}

/**
 * @brief Helper function to check memory allocation.
 */
template<typename T>
bool check_alloc(T* ptr, const string& name) {
    if (ptr == nullptr) {
        cerr << "Error: _mm_malloc failed for " << name << endl;
        return false;
    }
    return true;
}


// === Main Program ===

int main() {
    
    // --- 1. Allocate Aligned Memory ---
    float* input = (float*)_mm_malloc(INPUT_SIZE * sizeof(float), ALIGNMENT);
    float* weights_ih_t = (float*)_mm_malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float), ALIGNMENT);
    float* bias_h = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* hidden_serial = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* hidden_parallel = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);

    float* weights_ho_t = (float*)_mm_malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* bias_o = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);
    float* output_serial = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);
    float* output_parallel = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);

    if (!check_alloc(input, "input") || !check_alloc(weights_ih_t, "weights_ih_t") ||
        !check_alloc(bias_h, "bias_h") || !check_alloc(hidden_serial, "hidden_serial") ||
        !check_alloc(hidden_parallel, "hidden_parallel") ||
        !check_alloc(weights_ho_t, "weights_ho_t") || !check_alloc(bias_o, "bias_o") ||
        !check_alloc(output_serial, "output_serial") || !check_alloc(output_parallel, "output_parallel")) {
        return -1; 
    }

    // --- 2. Load Data from Binary Files ---
    cout << "Loading data from .bin files..." << endl;
    if (!read_from_file("data/input.bin", input, INPUT_SIZE) ||
        !read_from_file("data/weights_ih_t.bin", weights_ih_t, HIDDEN_SIZE * INPUT_SIZE) ||
        !read_from_file("data/bias_h.bin", bias_h, HIDDEN_SIZE) ||
        !read_from_file("data/weights_ho_t.bin", weights_ho_t, OUTPUT_SIZE * HIDDEN_SIZE) ||
        !read_from_file("data/bias_o.bin", bias_o, OUTPUT_SIZE)) {
        cerr << "Failed to load all data files. Exiting." << endl;
        return -1;
    }
    cout << "Data loaded successfully." << endl;

    // --- 3. Run Serial Execution and Timing ---
    cout << "\nRunning Serial Forward Pass..." << endl;
    
    unsigned long long start_serial, end_serial;
    start_serial = __rdtsc();
    
    forward_layer_serial(hidden_serial, input, weights_ih_t, bias_h, INPUT_SIZE, HIDDEN_SIZE);
    forward_layer_serial(output_serial, hidden_serial, weights_ho_t, bias_o, HIDDEN_SIZE, OUTPUT_SIZE);
    
    end_serial = __rdtsc();
    unsigned long long time_serial = end_serial - start_serial;
    

    // --- 4. Run Parallel (SIMD) Execution and Timing ---
    cout << "Running Parallel (SIMD) Forward Pass..." << endl;

    unsigned long long start_parallel, end_parallel;
    start_parallel = __rdtsc();

    forward_layer_parallel(hidden_parallel, input, weights_ih_t, bias_h, INPUT_SIZE, HIDDEN_SIZE);
    forward_layer_parallel(output_parallel, hidden_parallel, weights_ho_t, bias_o, HIDDEN_SIZE, OUTPUT_SIZE);

    end_parallel = __rdtsc();
    unsigned long long time_parallel = end_parallel - start_parallel;

    
    // --- 5. Verification ---
    cout << "\n--- Verification ---" << endl;
    bool match = true;
    // We compare the raw memory of the output arrays
    if (memcmp(output_serial, output_parallel, OUTPUT_SIZE * sizeof(float)) != 0) {
        match = false;
    }
    
    if(match) {
        cout << "SUCCESS: Serial and Parallel outputs match!" << endl;
    } else {
        cout << "FAILURE: Outputs do not match." << endl;
        // Optionally print both outputs for debugging
        cout << "Serial:" << endl;
        for(int i=0; i<OUTPUT_SIZE; ++i) cout << "  [" << i << "] = " << output_serial[i] << endl;
        cout << "Parallel:" << endl;
        for(int i=0; i<OUTPUT_SIZE; ++i) cout << "  [" << i << "] = " << output_parallel[i] << endl;
    }


    // --- 6. Display Results ---
    cout << "\n--- Performance Results ---" << endl;
    cout << "Serial Time (clocks):   " << time_serial << endl;
    cout << "Parallel Time (clocks): " << time_parallel << endl;
    cout << fixed << setprecision(2);
    cout << "Speedup:                " << (double)time_serial / time_parallel << "x" << endl;


    // --- 7. Free Aligned Memory ---
    _mm_free(input);
    _mm_free(weights_ih_t);
    _mm_free(bias_h);
    _mm_free(hidden_serial);
    _mm_free(hidden_parallel);
    _mm_free(weights_ho_t);
    _mm_free(bias_o);
    _mm_free(output_serial);
    _mm_free(output_parallel);

    return 0;
}