#include <stdio.h>
#include <string.h>
#include <immintrin.h>

void print_int_vector(__m128i a, const char *type) {
    if (strcmp(type, "u8") == 0) {
        unsigned char v[16];
        _mm_storeu_si128((__m128i*)v, a);
        printf("u8: [");
        for (int i = 15; i >= 0; i--) printf("%3u%s", v[i], i ? ", " : "");
        printf("]\n");
    }
    else if (strcmp(type, "i8") == 0) {
        signed char v[16];
        _mm_storeu_si128((__m128i*)v, a);
        printf("i8: [");
        for (int i = 15; i >= 0; i--) printf("%4d%s", v[i], i ? ", " : "");
        printf("]\n");
    }
    else if (strcmp(type, "u16") == 0) {
        unsigned short v[8];
        _mm_storeu_si128((__m128i*)v, a);
        printf("u16: [");
        for (int i = 7; i >= 0; i--) printf("%5u%s", v[i], i ? ", " : "");
        printf("]\n");
    }
    else if (strcmp(type, "i16") == 0) {
        short v[8];
        _mm_storeu_si128((__m128i*)v, a);
        printf("i16: [");
        for (int i = 7; i >= 0; i--) printf("%6d%s", v[i], i ? ", " : "");
        printf("]\n");
    }
    else if (strcmp(type, "u32") == 0) {
        unsigned int v[4];
        _mm_storeu_si128((__m128i*)v, a);
        printf("u32: [");
        for (int i = 3; i >= 0; i--) printf("%10u%s", v[i], i ? ", " : "");
        printf("]\n");
    }
    else if (strcmp(type, "i32") == 0) {
        int v[4];
        _mm_storeu_si128((__m128i*)v, a);
        printf("i32: [");
        for (int i = 3; i >= 0; i--) printf("%11d%s", v[i], i ? ", " : "");
        printf("]\n");
    }
    else if (strcmp(type, "u64") == 0) {
        unsigned long long v[2];
        _mm_storeu_si128((__m128i*)v, a);
        printf("u64: [");
        for (int i = 1; i >= 0; i--) printf("%20llu%s", v[i], i ? ", " : "");
        printf("]\n");
    }
    else if (strcmp(type, "i64") == 0) {
        long long v[2];
        _mm_storeu_si128((__m128i*)v, a);
        printf("i64: [");
        for (int i = 1; i >= 0; i--) printf("%21lld%s", v[i], i ? ", " : "");
        printf("]\n");
    }
    else {
        printf("Unknown type specifier.\n");
    }
}

int main() {
    __m128i v = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    print_int_vector(v, "u8");
    print_int_vector(v, "i8");
    print_int_vector(v, "u16");
    print_int_vector(v, "i16");
    print_int_vector(v, "u32");
    print_int_vector(v, "i32");
    print_int_vector(v, "u64");
    print_int_vector(v, "i64");
    return 0;
}
