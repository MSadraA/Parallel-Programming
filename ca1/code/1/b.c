#include <stdio.h>
#include <cpuid.h>

int isHyperthreadingSupported() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return (edx & (1 << 28)) ? 1 : 0;
}

void getTopologyInfo(int *threadsPerCore, int *logicalPerPackage) {
    unsigned int eax, ebx, ecx, edx;
    unsigned int maxBasicLeaf = __get_cpuid_max(0, NULL);
    unsigned int topologyLeaf = (maxBasicLeaf >= 0x1F) ? 0x1F : ((maxBasicLeaf >= 0xB) ? 0xB : 0);
    *threadsPerCore = 1;
    *logicalPerPackage = 1;

    if (topologyLeaf == 0) return;

    for (int i = 0; i < 10; i++) {
        __get_cpuid_count(topologyLeaf, i, &eax, &ebx, &ecx, &edx);
        int levelType = (ecx >> 8) & 0xFF;
        if (ebx == 0 || levelType == 0) break;
        if (levelType == 1) *threadsPerCore = ebx & 0xFFFF;
        if (levelType == 2) *logicalPerPackage = ebx & 0xFFFF;
    }
}

int getLogicalProcessorCount() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    int logicalCount = (ebx >> 16) & 0xFF;
    return logicalCount;
}

int getPhysicalCoreCount() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid_count(4, 0, &eax, &ebx, &ecx, &edx);
    int cores = ((eax >> 26) & 0x3F) + 1;
    return cores;
}

const char* getHyperthreadingState() {
    int supported = isHyperthreadingSupported();
    int logical = getLogicalProcessorCount();
    int physical = getPhysicalCoreCount();

    if (!supported)
        return "Not Supported";
    if (logical > physical)
        return "ON";
    else
        return "OFF";
}

int main() {
    printf("Hyperthreading Supported: %s\n",
           isHyperthreadingSupported() ? "Yes" : "No");
    printf("Logical Core Count: %d\n", getLogicalProcessorCount());
    printf("Physical Core Count: %d\n", getPhysicalCoreCount());
    printf("Hyperthreading State: %s\n", getHyperthreadingState());
    return 0;
}
