#include <stdio.h>
#include <cpuid.h>
#include <string.h>

int isHyperthreadingSupported() {
    unsigned int eax, ebx, ecx, edx;

    __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    return (edx & (1 << 28)) ? 1 : 0;
}

int getLogicalCoreCount() {
    unsigned int eax, ebx, ecx, edx;

    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    int logicalCount = (ebx >> 16) & 0xff;
    
    return logicalCount;
}

int getPhysicalCoreCount() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid_count(4, 0, &eax, &ebx, &ecx, &edx);
    return ((eax >> 26) & 0x3F) + 1;
}


int main() {
    printf("Hyperthreading Supported: %s\n", isHyperthreadingSupported() ? "Yes" : "No");
    printf("Logical Core Count: %d\n", getLogicalCoreCount());
    printf("Physical Core Count: %d\n", getPhysicalCoreCount());

    return 0;
}