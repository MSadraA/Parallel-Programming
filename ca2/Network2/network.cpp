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
const int AVX_LANE_COUNT = 8; 


// === Activation Function ===
inline float activation_relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

// === Serial Implementation (Baseline) ===
void forward_layer_serial(
    float* output_vector,
    const float* input_vector,
    const float* weights_transposed, // Reads 'weights_xx_t.bin'
    const float* bias_vector,
    int num_inputs,
    int num_outputs
) {
    for (int j = 0; j < num_outputs; ++j) {
        float sum = 0.0f;
        const float* p_weight_row = weights_transposed + (j * num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            sum += input_vector[i] * p_weight_row[i];
        }
        output_vector[j] = activation_relu(sum + bias_vector[j]);
    }
}


// === [NEW] Parallel (SIMD) Implementation - Idea 3 ===

inline __m256 _mm256_relu_ps(__m256 v) {
    const __m256 v_zero = _mm256_setzero_ps();
    return _mm256_max_ps(v, v_zero);
}

void forward_layer_parallel_v2(
    float* output_vector,
    const float* input_vector,
    const float* weights_vsimd,
    const float* bias_vector,
    int num_inputs,
    int num_outputs
) {
    // Loop over output neurons in chunks of 8
    for (int j = 0; j < num_outputs; j += AVX_LANE_COUNT) {
        
        // Load the 8 biases for neurons j, j+1, ..., j+7
        __m256 v_sum = _mm256_load_ps(bias_vector + j);

        // This pointer advances through the V-SIMD weight matrix
        const float* p_weight_chunk = weights_vsimd + (j / AVX_LANE_COUNT) * (num_inputs * AVX_LANE_COUNT);
        
        // --- Serial Inner Loop (over inputs) ---
        // This loop is serial, but its operations are AVX
        for (int i = 0; i < num_inputs; ++i) {
            
            // 1. Load *one* input and broadcast it to all 8 lanes
            __m256 v_input = _mm256_set1_ps(input_vector[i]);
            
            // 2. Load 8 contiguous weights from the V-SIMD layout
            //    (w[j,i], w[j+1,i], ..., w[j+7,i])
            __m256 v_weights = _mm256_load_ps(p_weight_chunk + (i * AVX_LANE_COUNT));
            
            // 3. Fused Multiply-Add
            // v_sum[0] = v_sum[0] + (v_input[0] * v_weights[0])
            // v_sum[1] = v_sum[1] + (v_input[1] * v_weights[1])
            // ... (and all 8 lanes are identical: in[i])
            v_sum = _mm256_fmadd_ps(v_input, v_weights, v_sum);
        }
        
        // --- Apply Activation (ReLU) on all 8 sums at once ---
        v_sum = _mm256_relu_ps(v_sum);
        
        // --- Store 8 final results ---
        _mm256_store_ps(output_vector + j, v_sum);
    }
}


// === Helper Functions (File I/O) ===
// (Unchanged from before)
bool read_from_file(const string& filename, float* data, size_t num_elements) {
    FILE* f = fopen(filename.c_str(), "rb");
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
    float* bias_h = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* bias_o = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);
    
    // Serial data
    float* weights_ih_t = (float*)_mm_malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float), ALIGNMENT);
    float* weights_ho_t = (float*)_mm_malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* hidden_serial = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* output_serial = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);

    // Parallel (V-SIMD) data
    float* weights_ih_vsimd = (float*)_mm_malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float), ALIGNMENT);
    float* weights_ho_vsimd = (float*)_mm_malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* hidden_parallel = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* output_parallel = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);

    // (Add check_alloc calls here for all pointers...)
    if (!check_alloc(input, "input") || !check_alloc(bias_h, "bias_h") || !check_alloc(bias_o, "bias_o")) return -1;
    if (!check_alloc(weights_ih_t, "weights_ih_t") || !check_alloc(weights_ho_t, "weights_ho_t")) return -1;
    if (!check_alloc(hidden_serial, "hidden_serial") || !check_alloc(output_serial, "output_serial")) return -1;
    if (!check_alloc(weights_ih_vsimd, "weights_ih_vsimd") || !check_alloc(weights_ho_vsimd, "weights_ho_vsimd")) return -1;
    if (!check_alloc(hidden_parallel, "hidden_parallel") || !check_alloc(output_parallel, "output_parallel")) return -1;


    // --- 2. Load Data from Binary Files ---
    cout << "Loading data from .bin files..." << endl;
    if (!read_from_file("data/input.bin", input, INPUT_SIZE) ||
        !read_from_file("data/bias_h.bin", bias_h, HIDDEN_SIZE) ||
        !read_from_file("data/bias_o.bin", bias_o, OUTPUT_SIZE)) {
        return -1; // Failed to load common data
    }

    // Load Serial-Optimized weights
    if (!read_from_file("data/weights_ih_t.bin", weights_ih_t, HIDDEN_SIZE * INPUT_SIZE) ||
        !read_from_file("data/weights_ho_t.bin", weights_ho_t, OUTPUT_SIZE * HIDDEN_SIZE)) {
        cerr << "Failed to load _t.bin files for Serial. Exiting." << endl;
        return -1;
    }
    
    // Load Parallel-Optimized weights
    if (!read_from_file("data/weights_ih_vsimd.bin", weights_ih_vsimd, HIDDEN_SIZE * INPUT_SIZE) ||
        !read_from_file("data/weights_ho_vsimd.bin", weights_ho_vsimd, OUTPUT_SIZE * HIDDEN_SIZE)) {
        cerr << "Failed to load _vsimd.bin files for Parallel. Exiting." << endl;
        return -1;
    }
    cout << "Data loaded successfully." << endl;

    
    // --- 3. Run Serial Execution and Timing ---
    cout << "\nRunning Serial Forward Pass (Optimized Layout)..." << endl;
    
    unsigned long long start_serial, end_serial;
    start_serial = __rdtsc();
    
    forward_layer_serial(hidden_serial, input, weights_ih_t, bias_h, INPUT_SIZE, HIDDEN_SIZE);
    forward_layer_serial(output_serial, hidden_serial, weights_ho_t, bias_o, HIDDEN_SIZE, OUTPUT_SIZE);
    
    end_serial = __rdtsc();
    unsigned long long time_serial = end_serial - start_serial;
    

    // --- 4. Run Parallel (SIMD) Execution and Timing ---
    cout << "Running Parallel (V-SIMD) Forward Pass..." << endl;

    unsigned long long start_parallel, end_parallel;
    start_parallel = __rdtsc();

    forward_layer_parallel_v2(hidden_parallel, input, weights_ih_vsimd, bias_h, INPUT_SIZE, HIDDEN_SIZE);
    forward_layer_parallel_v2(output_parallel, hidden_parallel, weights_ho_vsimd, bias_o, HIDDEN_SIZE, OUTPUT_SIZE);

    end_parallel = __rdtsc();
    unsigned long long time_parallel = end_parallel - start_parallel;


    // --- 5. Verification ---
    cout << "\n--- Verification ---" << endl;
    
    // We must use a float-aware comparison due to potential tiny precision 
    // differences in floating-point math (e.g., (a+b)+c != a+(b+c))
    bool match = true;
    const float EPSILON = 1e-6f; // A small tolerance
    
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        if (fabs(output_serial[i] - output_parallel[i]) > EPSILON) {
            match = false;
            cout << "Mismatch at [" << i << "]: Serial=" << output_serial[i] << ", Parallel=" << output_parallel[i] << endl;
            break;
        }
    }
    
    if(match) {
        cout << "SUCCESS: Serial and Parallel outputs match within tolerance!" << endl;
    } else {
        cout << "FAILURE: Outputs do not match." << endl;
    }


    // --- 6. Display Results ---
    cout << "\n--- Performance Results (Idea 3) ---" << endl;
    cout << "Serial Time (clocks):   " << time_serial << endl;
    cout << "Parallel Time (clocks): " << time_parallel << endl;
    cout << fixed << setprecision(2);
    cout << "Speedup:                " << (double)time_serial / time_parallel << "x" << endl;


    // --- 7. Free Aligned Memory ---
    _mm_free(input); _mm_free(bias_h); _mm_free(bias_o);
    _mm_free(weights_ih_t); _mm_free(weights_ho_t);
    _mm_free(hidden_serial); _mm_free(output_serial);
    _mm_free(weights_ih_vsimd); _mm_free(weights_ho_vsimd);
    _mm_free(hidden_parallel); _mm_free(output_parallel);

    return 0;
}