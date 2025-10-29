#include <stdio.h>
#include <cpuid.h>
#include <string.h>

char* getVendor() {
    static char vendor[13];
    unsigned int eax, ebx, ecx, edx;

    __get_cpuid(0x0, &eax, &ebx, &ecx, &edx);

    *((unsigned int*)&vendor[0])  = ebx;
    *((unsigned int*)&vendor[4])  = edx;
    *((unsigned int*)&vendor[8])  = ecx;
    vendor[12] = '\0';

    return vendor;
}

int  getBaseFrequency(){
    unsigned int eax, ebx, ecx, edx;

    __get_cpuid(0x16, &eax, &ebx, &ecx, &edx);

    return eax;
}

char *getBrand() {
    static char brand[49];
    unsigned int eax, ebx, ecx, edx;

    for (int i = 0; i < 3; i++) {
        __get_cpuid(0x80000002 + i, &eax, &ebx, &ecx, &edx);
        memcpy(&brand[i * 16], &eax, 4);
        memcpy(&brand[i * 16 + 4], &ebx, 4);
        memcpy(&brand[i * 16 + 8], &ecx, 4);
        memcpy(&brand[i * 16 + 12], &edx, 4);
    }
    brand[48] = '\0';

    return brand;
}

int main() {
    unsigned int eax, ebx, ecx, edx;

    printf("CPU Vendor: %s\n", getVendor());
    printf("CPU Brand: %s\n", getBrand());
    printf("Base Frequency: %d MHz\n", getBaseFrequency());
}