import numpy as np
import os

# --- تعریف ساختار شبکه ---
INPUT_SIZE = 8
HIDDEN_SIZE = 16
OUTPUT_SIZE = 8

# --- تولید داده‌ها ---
# داده‌ها را با نوع float32 (معادل float در C++) تولید می‌کنیم
# استفاده از np.random.uniform(-1.0, 1.0, ...) یک استاندارد خوب برای شروع وزن‌ها است

print("Generating random data...")

# 1. لایه ورودی
# (8,)
input_vector = np.random.uniform(-1.0, 1.0, size=INPUT_SIZE).astype(np.float32)

# 2. وزن‌های ورودی به پنهان (از قبل ترانهاده شده)
# (16, 8) -> [num_outputs, num_inputs]
weights_ih_t = np.random.uniform(-1.0, 1.0, size=(HIDDEN_SIZE, INPUT_SIZE)).astype(np.float32)

# 3. بایاس لایه پنهان
# (16,)
bias_h = np.random.uniform(-1.0, 1.0, size=HIDDEN_SIZE).astype(np.float32)

# 4. وزن‌های پنهان به خروجی (از قبل ترانهاده شده)
# (8, 16) -> [num_outputs, num_inputs]
weights_ho_t = np.random.uniform(-1.0, 1.0, size=(OUTPUT_SIZE, HIDDEN_SIZE)).astype(np.float32)

# 5. بایاس لایه خروجی
# (8,)
bias_o = np.random.uniform(-1.0, 1.0, size=OUTPUT_SIZE).astype(np.float32)

# --- ذخیره در فایل‌های باینری ---
# از متد .tofile() برای ذخیره‌سازی خام باینری استفاده می‌کنیم
try:
    input_vector.tofile('data/input.bin')
    weights_ih_t.tofile('data/weights_ih_t.bin')
    bias_h.tofile('data/bias_h.bin')
    weights_ho_t.tofile('data/weights_ho_t.bin')
    bias_o.tofile('data/bias_o.bin')
    
    print("Successfully generated and saved all .bin files.")
    print(f"  - input.bin         : {input_vector.nbytes} bytes")
    print(f"  - weights_ih_t.bin  : {weights_ih_t.nbytes} bytes")
    print(f"  - bias_h.bin        : {bias_h.nbytes} bytes")
    print(f"  - weights_ho_t.bin  : {weights_ho_t.nbytes} bytes")
    print(f"  - bias_o.bin        : {bias_o.nbytes} bytes")

except Exception as e:
    print(f"An error occurred while writing files: {e}")