#include 	"stdio.h"
#include 	"ipp.h"
#include 	"x86intrin.h"

#define     VECTOR_SIZE      16000000

int main()
{
    float *v1;
    Ipp64u start, end;
	Ipp64u time1, time2;

    v1 = new float [VECTOR_SIZE];
    for (long i = 0; i < VECTOR_SIZE; i++)
		v1[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));

    float target = v1[14000000];

    start = ippGetCpuClocks();
    for (long i = 0; i<VECTOR_SIZE; i++)
        if (v1[i] == target){
            printf("Found %.2f in serial code!\n", target); 
            break;
        }
    end   = ippGetCpuClocks();
	time1 = end - start;

    __m128 vTarget, vData, vCmp;

    start = ippGetCpuClocks();
    vTarget = _mm_set1_ps(target);
    for(long int i = 0; i < VECTOR_SIZE; i += 4) {
        vData = _mm_loadu_ps(&v1[i]);
        vCmp  = _mm_cmpeq_ps(vData, vTarget);
        int mask = _mm_movemask_ps(vCmp);

        if(mask != 0){
            printf("Found %.2f in parallel code!\n", target);    
            break; 
        }
    }
    end   = ippGetCpuClocks();
	time2 = end - start;

    printf ("Serial Run time = %d \n", (Ipp32s) time1);
    printf ("Parallel Run time = %d \n", (Ipp32s) time2);
    printf ("\nSpeedup = %f\n\n", (float) (time1)/(float) time2);

    return 0;
}