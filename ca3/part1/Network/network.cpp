#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>      // For std::setprecision
#include <x86intrin.h>  // For __rdtsc()
#include <malloc.h>     // For _mm_malloc / _mm_free (Aligned allocation)
#include <cstdio>       // For fopen, fread, fclose
#include <string>       // For std::string
#include <cstring>      // For memcmp (verification)
#include <omp.h>


using namespace std;

// --- Network Structure Definitions ---
const int INPUT_SIZE = 4096;
const int HIDDEN_SIZE = 8192; 
const int OUTPUT_SIZE = 4096; 

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

void forward_layer_openmp(float* output_vector, const float* input_vector, 
                          const float* weights_transposed, const float* bias_vector, 
                          int num_inputs, int num_outputs) {
    int j;
    #pragma omp parallel for schedule(static) private(j) \
            shared(output_vector, input_vector, weights_transposed, bias_vector, num_inputs, num_outputs)
    for (j = 0; j < num_outputs; ++j) {
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

void forward_layer_simd(
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

void forward_layer_omp_simd(float* output_vector, const float* input_vector, 
                            const float* weights_vsimd, const float* bias_vector, 
                            int num_inputs, int num_outputs) {
    int j;
    #pragma omp parallel for schedule(static) private(j) \
            shared(output_vector, input_vector, weights_vsimd, bias_vector, num_inputs, num_outputs)
    for (j = 0; j < num_outputs; j += AVX_LANE_COUNT) {
        __m256 v_sum = _mm256_load_ps(bias_vector + j);
        const float* p_weight_chunk = weights_vsimd + (j / AVX_LANE_COUNT) * (num_inputs * AVX_LANE_COUNT);
        
        for (int i = 0; i < num_inputs; ++i) {
            __m256 v_input = _mm256_set1_ps(input_vector[i]);
            __m256 v_weights = _mm256_load_ps(p_weight_chunk + (i * AVX_LANE_COUNT));
            v_sum = _mm256_fmadd_ps(v_input, v_weights, v_sum);
        }
        v_sum = _mm256_relu_ps(v_sum);
        _mm256_store_ps(output_vector + j, v_sum);
    }
}

// === Helper Functions (File I/O) ===
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


// --- Verification Helper ---
void verify(const float* ref, const float* target, int size, const char* name) {
    float max_diff = 0.0f;
    for(int i=0; i<size; i++) {
        float diff = fabs(ref[i] - target[i]);
        if(diff > max_diff) max_diff = diff;
    }
    if (max_diff < 1e-2f) 
        printf("Verify %-15s: PASSED (Max Diff: %.1e)\n", name, max_diff);
    else 
        printf("Verify %-15s: FAILED (Max Diff: %.1e)\n", name, max_diff);
}

const int NUM_REPEATS = 1000;

int main() {
    // 1. Allocation
    float* input = (float*)_mm_malloc(INPUT_SIZE * sizeof(float), ALIGNMENT);
    float* bias_h = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* bias_o = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);
    
    // Weights
    float* w_ih_t = (float*)_mm_malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float), ALIGNMENT);
    float* w_ho_t = (float*)_mm_malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* w_ih_v = (float*)_mm_malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float), ALIGNMENT);
    float* w_ho_v = (float*)_mm_malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    
    // Outputs Buffers
    float* h_ser = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* o_ser = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);
    
    float* h_simd = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* o_simd = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);
    
    float* h_omp = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* o_omp = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);

    float* h_omp_simd = (float*)_mm_malloc(HIDDEN_SIZE * sizeof(float), ALIGNMENT);
    float* o_omp_simd = (float*)_mm_malloc(OUTPUT_SIZE * sizeof(float), ALIGNMENT);

    // 2. Load Data
    bool ok = true;
    ok &= read_from_file("data/input.bin", input, INPUT_SIZE);
    ok &= read_from_file("data/bias_h.bin", bias_h, HIDDEN_SIZE);
    ok &= read_from_file("data/bias_o.bin", bias_o, OUTPUT_SIZE);
    ok &= read_from_file("data/weights_ih_t.bin", w_ih_t, HIDDEN_SIZE * INPUT_SIZE);
    ok &= read_from_file("data/weights_ho_t.bin", w_ho_t, OUTPUT_SIZE * HIDDEN_SIZE);
    ok &= read_from_file("data/weights_ih_vsimd.bin", w_ih_v, HIDDEN_SIZE * INPUT_SIZE);
    ok &= read_from_file("data/weights_ho_vsimd.bin", w_ho_v, OUTPUT_SIZE * HIDDEN_SIZE);

    if (!ok) { cerr << "Error loading files. Run python script." << endl; return -1; }

    cout << "Network: " << INPUT_SIZE << " -> " << HIDDEN_SIZE << " -> " << OUTPUT_SIZE << endl;
    cout << "Threads: " << omp_get_max_threads() << endl << endl;

    unsigned long long start, t_ser, t_simd, t_omp, t_omp_simd;

    // --- 1. Serial ---
    start = __rdtsc();
        forward_layer_serial(h_ser, input, w_ih_t, bias_h, INPUT_SIZE, HIDDEN_SIZE);
        forward_layer_serial(o_ser, h_ser, w_ho_t, bias_o, HIDDEN_SIZE, OUTPUT_SIZE);
    t_ser = __rdtsc() - start;
    cout << "Serial Clocks:       " << t_ser << endl;

    // --- 2. SIMD (Single Core) ---
    start = __rdtsc();
        forward_layer_simd(h_simd, input, w_ih_v, bias_h, INPUT_SIZE, HIDDEN_SIZE);
        forward_layer_simd(o_simd, h_simd, w_ho_v, bias_o, HIDDEN_SIZE, OUTPUT_SIZE);
    t_simd = __rdtsc() - start;
    cout << "SIMD Clocks:         " << t_simd << endl;

    // --- 3. OpenMP (Multi Core, No SIMD) ---
    start = __rdtsc();
        forward_layer_openmp(h_omp, input, w_ih_t, bias_h, INPUT_SIZE, HIDDEN_SIZE);
        forward_layer_openmp(o_omp, h_omp, w_ho_t, bias_o, HIDDEN_SIZE, OUTPUT_SIZE);
    t_omp = __rdtsc() - start;
    cout << "OpenMP Clocks:       " << t_omp << endl;

    // --- 4. OpenMP + SIMD ---
    start = __rdtsc();
        forward_layer_omp_simd(h_omp_simd, input, w_ih_v, bias_h, INPUT_SIZE, HIDDEN_SIZE);
        forward_layer_omp_simd(o_omp_simd, h_omp_simd, w_ho_v, bias_o, HIDDEN_SIZE, OUTPUT_SIZE);
    t_omp_simd = __rdtsc() - start;
    cout << "OpenMP+SIMD Clocks:  " << t_omp_simd << endl;

    cout << "\n--------------------------------" << endl;
    cout << "Speedup (SIMD vs Serial):        " << fixed << setprecision(2) << (double)t_ser / t_simd << "x" << endl;
    cout << "Speedup (OpenMP vs Serial):      " << (double)t_ser / t_omp << "x" << endl;
    cout << "Speedup (OpenMP+SIMD vs Serial): " << (double)t_ser / t_omp_simd << "x" << endl;
    cout << "--------------------------------" << endl;

    // --- Verification ---
    verify(o_ser, o_simd, OUTPUT_SIZE, "SIMD");
    verify(o_ser, o_omp, OUTPUT_SIZE, "OpenMP");
    verify(o_ser, o_omp_simd, OUTPUT_SIZE, "OpenMP+SIMD");

    // Cleanup
    _mm_free(input); _mm_free(bias_h); _mm_free(bias_o);
    _mm_free(w_ih_t); _mm_free(w_ho_t); _mm_free(w_ih_v); _mm_free(w_ho_v);
    _mm_free(h_ser); _mm_free(o_ser); _mm_free(h_simd); _mm_free(o_simd);
    _mm_free(h_omp); _mm_free(o_omp); _mm_free(h_omp_simd); _mm_free(o_omp_simd);

    return 0;
}