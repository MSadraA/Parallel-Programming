#include 	"stdio.h"
#include 	"stdlib.h"
#include 	"ipp.h"
#include 	"x86intrin.h"
#include 	"time.h"

#define		VECTOR_SIZE		1600000
#define		MIN_VALUE		1000
#define		MAX_VALUE		4000

int main (void)
{
	Ipp64u start, end;
	Ipp64u time1, time2;

	float *v1, *v2;
	v1 = new float [VECTOR_SIZE];
	v2 = new float [VECTOR_SIZE];
	

	// Initialize vectors with random numbers
	srand(time(0));
	for (long i = 0; i < VECTOR_SIZE; i++)
		v1[i] = (float) ((rand()%(MAX_VALUE-MIN_VALUE+1))+MIN_VALUE);

	// Inner product, Scalar implementation
	start = ippGetCpuClocks();
	for (long i = 0; i < VECTOR_SIZE; i++)
		v2[i] = (v1[i] - MIN_VALUE)/(MAX_VALUE - MIN_VALUE);
	end   = ippGetCpuClocks();
	time1 = end - start;

	// Inner product, Vector implementation
	__m128 vMin = _mm_set1_ps(MIN_VALUE);
    __m128 vRange = _mm_set1_ps(MAX_VALUE - MIN_VALUE);
	start = ippGetCpuClocks();
    for (int i = 0; i < VECTOR_SIZE; i += 4) {
        __m128 vData = _mm_loadu_ps(&v1[i]);
        vData = _mm_sub_ps(vData, vMin);
        vData = _mm_div_ps(vData, vRange);
        _mm_storeu_ps(&v2[i], vData);
    }
	end   = ippGetCpuClocks();
	time2 = end - start;

	printf ("Serial Run time = %d \n", (Ipp32s) time1);
	printf ("Parallel Run time = %d \n", (Ipp32s) time2);
	printf ("Speedup = %f\n\n", (float) (time1)/(float) time2);

	return 0;
}
