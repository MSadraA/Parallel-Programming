#include <stdio.h>
#include <cpuid.h>

int isMMXSupported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (edx & (1 << 23)) ? 1 : 0;
}

int isSSESupported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (edx & (1 << 25)) ? 1 : 0;
}

int isSSE2Supported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (edx & (1 << 26)) ? 1 : 0;
}

int isSSE3Supported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1 << 0)) ? 1 : 0;
}

int isSSSE3Supported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1 << 9)) ? 1 : 0;
}

int isSSE41Supported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1 << 19)) ? 1 : 0;
}

int isSSE42Supported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1 << 20)) ? 1 : 0;
}

int isAVXSupported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (ecx & (1 << 28)) ? 1 : 0;
}

int isAVX2Supported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & (1 << 5)) ? 1 : 0;
}

void printSIMDSupport() {
    printf("Supported SIMD Architectures:\n");
    if (isMMXSupported())   printf("  MMX\n");
    if (isSSESupported())   printf("  SSE\n");
    if (isSSE2Supported())  printf("  SSE2\n");
    if (isSSE3Supported())  printf("  SSE3\n");
    if (isSSSE3Supported()) printf("  SSSE3\n");
    if (isSSE41Supported()) printf("  SSE4.1\n");
    if (isSSE42Supported()) printf("  SSE4.2\n");
    if (isAVXSupported())   printf("  AVX\n");
    if (isAVX2Supported())  printf("  AVX2\n");
}

int main() {
    printSIMDSupport();
    return 0;
}
