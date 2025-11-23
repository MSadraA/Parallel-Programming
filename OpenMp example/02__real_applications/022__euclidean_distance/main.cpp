#include 	"stdio.h"
#include 	"ipp.h"
#include 	"math.h"
#include 	"x86intrin.h"

#define		VECTOR_SIZE		65536

int main (void)
{
	Ipp64u start, end;
	Ipp64u time1, time2;

	float fSTmpRes[4];
	float fSRes;
	float fVRes;

	float *v1, *v2;
	v1 = new float [VECTOR_SIZE];
	v2 = new float [VECTOR_SIZE];

	if (!v1 || !v2) {
		printf ("Memory allocation error!!\n");
		return 1;
	}
	// Initialize vectors with random numbers
	for (long i = 0; i < VECTOR_SIZE; i++)
	{
		v1[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));
		v2[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));
	}

	// Inner product, Scalar implementation
	start = ippGetCpuClocks();
	fSTmpRes[0] = fSTmpRes[1] = fSTmpRes[2] = fSTmpRes[3] = 0.0;
	for (long i = 0; i < VECTOR_SIZE; i+=4)
		fSTmpRes[0] += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	for (long i = 0; i < VECTOR_SIZE; i+=4)
		fSTmpRes[1] += (v1[i+1] - v2[i+1]) * (v1[i+1] - v2[i+1]);
	for (long i = 0; i < VECTOR_SIZE; i+=4)
		fSTmpRes[2] += (v1[i+2] - v2[i+2]) * (v1[i+2] - v2[i+2]);
	for (long i = 0; i < VECTOR_SIZE; i+=4)
		fSTmpRes[3] += (v1[i+3] - v2[i+3]) * (v1[i+3] - v2[i+3]);
	fVRes = (fSTmpRes[0] + fSTmpRes[1]) + (fSTmpRes[2] + fSTmpRes[3]);
	fSRes = (float) sqrt(fVRes);
	end   = ippGetCpuClocks();
	time1 = end - start;

	// Inner product, Vector implementation
	__m128 va, vb, diff, sum;
	
	start = ippGetCpuClocks();
	sum = _mm_setzero_ps();
    for(long int i = 0; i < VECTOR_SIZE; i += 4) {
        va = _mm_loadu_ps(&v1[i]);
        vb = _mm_loadu_ps(&v2[i]);
        diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
	sum = _mm_hadd_ps (sum, sum);
	sum = _mm_hadd_ps (sum, sum);
	fVRes = (float) sqrt(_mm_cvtss_f32 (sum));
	end   = ippGetCpuClocks();
	time2 = end - start;

	printf ("\nThe serial result is   = %f\n", fSRes);
	printf ("Serial Run time = %d \n", (Ipp32s) time1);
	printf ("The parallel result is = %f\n", fVRes);
	printf ("Parallel Run time = %d \n", (Ipp32s) time2);
	printf ("Speedup = %f\n\n", (float) (time1)/(float) time2);

	return 0;
}
