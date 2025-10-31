#include 	"stdio.h"
#include 	"ipp.h"
#include 	"math.h"
#include 	"time.h"
#include 	"x86intrin.h"

#define		VECTOR_SIZE		16000000

int main()
{
	Ipp64u start, end, time1, time2;

	float *w, *grad;
	float *n_w_ser, *n_w_par;
	const float learning_rate = 0.01;
	w       = new float [VECTOR_SIZE];
	grad    = new float [VECTOR_SIZE];
	n_w_ser = new float [VECTOR_SIZE];
	n_w_par = new float [VECTOR_SIZE];

	for (long int i=0; i<VECTOR_SIZE; i++){
		w[i]    = (float) ((rand() % 10) + 1);
		grad[i] = (-1.0 + (float)rand()/(RAND_MAX)) / 5.0;
	}

	start = ippGetCpuClocks();
	for (long int i=0; i<VECTOR_SIZE; i++){
		n_w_ser[i] = w[i] - learning_rate * grad[i];
	}
	end   = ippGetCpuClocks();
	time1 = end - start;

	__m128 vLR, vW, vGrad;
	start = ippGetCpuClocks();
	vLR = _mm_set1_ps(learning_rate);
    for(long int i=0; i<VECTOR_SIZE; i+=4){
        vW = _mm_loadu_ps(&w[i]);
		vGrad = _mm_loadu_ps(&grad[i]);
		vW = _mm_sub_ps(vW, _mm_mul_ps(vLR, vGrad));
		_mm_storeu_ps(&n_w_par[i], vW);
    }
	end   = ippGetCpuClocks();
	time2 = end - start;

	for(int i=0;i<VECTOR_SIZE;i++)
		if (n_w_ser[i] != n_w_par[i])
			printf("Failed!");

	printf ("Serial Run time = %d \n", (Ipp32s) time1);
	printf ("Parallel Run time = %d \n", (Ipp32s) time2);
	printf ("Speedup = %f\n\n", (float) (time1)/(float) time2);

	return 0;
}


// int main (void)
// {
// 	Ipp64u start, end;
// 	Ipp64u time1, time2;

// 	float fSTmpRes[4];
// 	float fSRes;
// 	float fVRes;

// 	float *w, *new_w, *new_w2, *grad;
// 	w      = new float [VECTOR_SIZE];
// 	new_w  = new float [VECTOR_SIZE];
// 	new_w2 = new float [VECTOR_SIZE];
// 	grad   = new float [VECTOR_SIZE];

// 	float learning_rate = 0.01;

// 	if (!w || !grad) {
// 		printf ("Memory allocation error!!\n");
// 		return 1;
// 	}
	
// 	// Initialize weight and grad with random numbers
// 	srand(time(0));
// 	for (long i = 0; i < VECTOR_SIZE; i++){
// 		w[i]    = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000));
// 		grad[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
// 	}

// 	// Serial implementation
// 	start = ippGetCpuClocks();
// 	for (long int i=0; i<VECTOR_SIZE; i++)
// 		new_w[i] = w[i] - learning_rate * grad[i];
// 	end   = ippGetCpuClocks();
// 	time1 = end - start;
	

// 	// Vector implementation
// 	__m128 w_vec, grad_vec, new_w_vec, lr_vec;

// 	start = ippGetCpuClocks();
// 	lr_vec = _mm_set1_ps(learning_rate);
// 	for (long int i=0; i<VECTOR_SIZE; i=i+4){
// 		w_vec     = _mm_loadu_ps(&w[i]);
// 		grad_vec  = _mm_loadu_ps(&grad[i]);
// 		new_w_vec = _mm_sub_ps(w_vec, _mm_mul_ps(lr_vec, grad_vec));
// 		_mm_storeu_ps(new_w2, new_w_vec);
// 	}
// 	end   = ippGetCpuClocks();
// 	time2 = end - start;

// 	for (long int i=0; i<VECTOR_SIZE; i++){
// 		printf("%f\t", w[i]);
// 	}
// 	printf("\n\n");
// 	for (long int i=0; i<VECTOR_SIZE; i++){
// 		printf("%f\t", new_w[i]);
// 	}	
// 	printf("\n\n");
// 	for (long int i=0; i<VECTOR_SIZE; i++){
// 		printf("%f\t", new_w2[i]);
// 	}

// 	printf("Serial and parallel implementations are matched!\n\n");

// 	printf ("Serial Run time = %d \n", (Ipp32s) time1);
// 	printf ("Parallel Run time = %d \n", (Ipp32s) time2);
// 	printf ("Speedup = %f\n\n", (float) (time1)/(float) time2);

// 	return 0;
// }
