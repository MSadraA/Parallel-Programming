#include <stdio.h>
#include <cpuid.h>

int getCacheType(unsigned int eax) {
    return eax & 0x1F;
}

int getCacheLevel(unsigned int eax) {
    return (eax >> 5) & 0x7;
}

int getCacheWays(unsigned int ebx) {
    return ((ebx >> 22) & 0x3FF) + 1;
}

int getCachePartitions(unsigned int ebx) {
    return ((ebx >> 12) & 0x3FF) + 1;
}

int getCacheLineSize(unsigned int ebx) {
    return (ebx & 0xFFF) + 1;
}

int getCacheSets(unsigned int ecx) {
    return ecx + 1;
}

int getCacheSizeKB(unsigned int eax, unsigned int ebx, unsigned int ecx) {
    int ways = getCacheWays(ebx);
    int partitions = getCachePartitions(ebx);
    int lineSize = getCacheLineSize(ebx);
    int sets = getCacheSets(ecx);
    return (ways * partitions * lineSize * sets) / 1024;
}

void printCacheInfo() {
    unsigned int eax, ebx, ecx, edx;
    int i = 0;

    while (1) {
        __get_cpuid_count(4, i, &eax, &ebx, &ecx, &edx);
        int type = getCacheType(eax);
        if (type == 0) break;

        int level = getCacheLevel(eax);
        int sizeKB = getCacheSizeKB(eax, ebx, ecx);

        printf("Cache Level %d: ", level);
        if (type == 1) printf("Data Cache, ");
        else if (type == 2) printf("Instruction Cache, ");
        else if (type == 3) printf("Unified Cache, ");
        else printf("Unknown Type, ");
        printf("Size = %d KB\n", sizeKB);
        i++;
    }
}

int main() {
    printCacheInfo();
    return 0;
}
