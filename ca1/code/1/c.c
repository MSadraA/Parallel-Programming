#include <stdio.h>
#include <cpuid.h>

int getMaxFrequency() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(0x16, &eax, &ebx, &ecx, &edx);
    return ebx;
}

int main() {
    unsigned int maxFreq = getMaxFrequency();
    if (maxFreq)
        printf("Maximum Frequency: %u MHz (%.2f GHz)\n", maxFreq, maxFreq / 1000.0);
    else
        printf("Leaf 0x16 not supported on this CPU.\n");
    return 0;
}
