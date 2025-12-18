#
# generate_data_v2.py
#
import numpy as np
import os

# --- Network Structure ---
INPUT_SIZE = 4096
HIDDEN_SIZE = 8192
OUTPUT_SIZE = 4096
AVX_LANE_COUNT = 8 # We process 8 floats at a time

# --- Helper function for the new V-SIMD layout ---
def save_vsimd_layout(weights_transposed, filename, num_outputs, num_inputs):
    """
    Saves weights in a V-SIMD (Vertical SIMD) friendly layout.
    
    Old Layout (Transposed): [j, i]
    (n0_i0, n0_i1, ...), (n1_i0, n1_i1, ...), ...
    
    New V-SIMD Layout: Groups 8 neurons together.
    (n0_i0, n1_i0, ..., n7_i0), (n0_i1, n1_i1, ..., n7_i1), ...
    """
    
    # Create the new array to hold re-ordered data
    vsimd_data = np.zeros(num_outputs * num_inputs, dtype=np.float32)
    idx = 0
    
    # Loop over output neurons in chunks of 8
    for j_chunk_start in range(0, num_outputs, AVX_LANE_COUNT):
        # Loop over each input
        for i in range(num_inputs):
            # For this input 'i', get the weights for the next 8 neurons
            for j in range(j_chunk_start, j_chunk_start + AVX_LANE_COUNT):
                # Ensure we don't go out of bounds if num_outputs isn't a multiple of 8
                if j < num_outputs:
                    vsimd_data[idx] = weights_transposed[j, i]
                # else: data remains 0 (padding)
                idx += 1
                
    # Save the re-ordered data
    vsimd_data.tofile(filename)


# --- 1. Generate Standard Data ---
print("Generating random data...")

# (8,)
input_vector = np.random.uniform(-1.0, 1.0, size=INPUT_SIZE).astype(np.float32)

# (16, 8) -> [num_outputs, num_inputs]
weights_ih_t = np.random.uniform(-1.0, 1.0, size=(HIDDEN_SIZE, INPUT_SIZE)).astype(np.float32)

# (16,)
bias_h = np.random.uniform(-1.0, 1.0, size=HIDDEN_SIZE).astype(np.float32)

# (8, 16) -> [num_outputs, num_inputs]
weights_ho_t = np.random.uniform(-1.0, 1.0, size=(OUTPUT_SIZE, HIDDEN_SIZE)).astype(np.float32)

# (8,)
bias_o = np.random.uniform(-1.0, 1.0, size=OUTPUT_SIZE).astype(np.float32)

# --- 2. Save Binary Files ---
try:
    # Save common files
    input_vector.tofile('data/input.bin')
    bias_h.tofile('data/bias_h.bin')
    bias_o.tofile('data/bias_o.bin')
    
    # --- Save weights in TWO formats ---
    
    # Format 1: Transposed (for Serial)
    weights_ih_t.tofile('data/weights_ih_t.bin')
    weights_ho_t.tofile('data/weights_ho_t.bin')
    print("Saved _t.bin files (for Serial).")

    # Format 2: V-SIMD (for Parallel)
    save_vsimd_layout(weights_ih_t, 'data/weights_ih_vsimd.bin', HIDDEN_SIZE, INPUT_SIZE)
    save_vsimd_layout(weights_ho_t, 'data/weights_ho_vsimd.bin', OUTPUT_SIZE, HIDDEN_SIZE)
    print("Saved _vsimd.bin files (for Parallel).")
    
    print("\nSuccessfully generated all .bin files.")

except Exception as e:
    print(f"An error occurred while writing files: {e}")