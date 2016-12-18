#!/usr/bin/env python
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy as np

kernelCUDA = """
#include <stdio.h>

// finding max with float using atomicCAS
__device__ float atomicMaxf(float* address, float val) {

	int *address_as_int = (int*)address;
	int old = *address_as_int;
	int tmp;

	while (val > __int_as_float(old)) {
		tmp = old;
		old = atomicCAS(address_as_int, tmp, __float_as_int(val));
	}
	return __int_as_float(old);
}

__global__ void similarity(float* x, float* y, float* z, float* s) { //DON'T CHANGE

	unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col >= %(N)s || row >= %(N)s)
		return;

	float dist = 0;
	float xi = x[row]-x[col];
	float yi = y[row]-y[col];
	float zi = z[row]-z[col];

	dist = -(xi*xi + yi*yi + zi*zi);

	s[row*%(N)s+col] = dist;
	__syncthreads();
}

// Calculate the preference with a single block, 1024 threads
// Should be median but this is a mean
__global__ void preference(float* S) { //DON'T CHANGE
	unsigned int i;
    unsigned int j;
    unsigned int k;
    float P_priv = 0.0;
    __shared__ float P_sh[1];

	if(threadIdx.x == 0)
		P_sh[0] = 0.0;

	// Calculate the mean by adding the lower triangle of S, which is symmetrical
    for (i=0; i<%(N)s; i++) {
		for (j=0; j<=i/blockDim.x; j++) {
			k = threadIdx.x + blockDim.x * j;
			if (k < i)
				P_priv += S[i*%(N)s + k];
		}
	}
    atomicAdd(&P_sh[0], 2*P_priv/%(N)s/%(N)s);
    __syncthreads();

    for (j=0; j<%(N)s; j+=blockDim.x) {
		S[(j + threadIdx.x)*(%(N)s + 1)] = P_sh[0];
    }
}

// Calculate responsibilities
__global__ void responsibilities(float* S, float* R, float* A, float *AS) { //TODO: CHANGE ACCESS FROM A AS COLUMN-WISE
	unsigned int i;
	unsigned int j;
	float temp;
	float pmax[2] = {%(NEG_MAX)s, %(NEG_MAX)s};
	unsigned int blk_os = blockIdx.x * %(N)s;
	__shared__ float max[2];
	__shared__ unsigned int max_idx;

	// default maximum to minimum representable float
	if (threadIdx.x <= 1)
		max[threadIdx.x] = %(NEG_MAX)s;
	__syncthreads();

	// AS = A + S
	for (i=0; i<%(N)s; i+=blockDim.x) {
		j = blk_os + i + threadIdx.x;
		temp = A[(i+threadIdx.x)*%(N)s+blockIdx.x] + S[j];
		AS[j] = temp;
		pmax[0] = (pmax[0] >= temp)*pmax[0] + (pmax[0] < temp)*temp;
	}
	atomicMaxf(&max[0], pmax[0]); // find maximum
	__syncthreads();

	pmax[0] = max[0];
	// Set max(AS) = -Inf then find next max
	for (i=0; i<%(N)s; i+=blockDim.x) {
		j = blk_os + i + threadIdx.x;
		temp = AS[j];
		if (temp == pmax[0]) {
		 	temp = %(NEG_MAX)s;
			AS[j] = temp;
			max_idx = j;
		}
		pmax[1] = (pmax[1] >= temp)*pmax[1] + (pmax[1] < temp)*temp;
	}
	atomicMaxf(&max[1], pmax[1]); // find next max
	__syncthreads();

	pmax[1] = max[1];
	// Apply damping and get new responsibility
	for (i=0; i<%(N)s; i+=blockDim.x) {
		j = blk_os + i + threadIdx.x;
		float old_R = R[j];
		if (j == max_idx)
			temp = S[j] - pmax[1];
		else
			temp = S[j] - pmax[0];
		R[j] = (1-%(DAMP)s) * temp + %(DAMP)s * old_R;
	}
}

// Calculate availabilities
__global__ void availabilities(float* A, float* R, float* RP) { //CHANGE ACCESS FROM R AS COLUMN-WISE
	unsigned int i;
	unsigned int j;
	unsigned int diag = blockIdx.x*(%(N)s+1);
	float rp;
	float a;
	float a_old;
	float sum_priv = 0.0;
	unsigned int blk_os = blockIdx.x * %(N)s;
	__shared__ float sum[1];

	if(threadIdx.x == 0)
		sum[0] = 0.0;
	__syncthreads();
	// Get elementwise maximum of R and calculate sum
	for (i=0; i<%(N)s; i+=blockDim.x) {
		j = blk_os + i + threadIdx.x;
		rp = R[(i+threadIdx.x)*%(N)s+blockIdx.x];
		if (rp < 0.0 && j != diag)
			rp = 0.0;
		RP[j] = rp;
		sum_priv += rp;
	}
	atomicAdd(&sum[0], sum_priv);
	__syncthreads();
	sum_priv = sum[0];

	// Calculate new availability
	for (i=0; i<%(N)s; i+=blockDim.x) {
		j = blk_os + i + threadIdx.x;
		a = sum_priv - RP[j];
		if (a > 0.0 && j != diag)
			a = 0.0;
		a_old = A[j];
		A[j] = (1-%(DAMP)s)*a + %(DAMP)s*a_old; //dampen
	}
}

// Check for convergence on the diagonal of A & R
// Use 1 block of 1024 threads
// Note the converged flag only tells CPU to stop iterating.
// CPU will not call this if MAXITS hit, so maxits checks are removed
__global__ void convergence(float* A, float* R, bool* E, unsigned int* e,
 	unsigned int* se, unsigned int iteration, bool* converged) { //TODO: UPDATE INDEXING OF A & R TO DIAGONAL. SEE PREFERENCE KERNEL.
	unsigned int i;
	unsigned int j;
	unsigned int k;
	bool E_priv;
	unsigned int e_sum = 0;
	unsigned int temp = 0;
	__shared__ unsigned int K[1]; //sum of E
	__shared__ unsigned int conv_sum[1];

	if (threadIdx.x == 0) {
		K[0] = 0;
		conv_sum[0] = 0;
	}
	__syncthreads();

	// E is bool vector of whether diagonal (A+R) > 0
	for (i=0; i<%(N)s; i+=blockDim.x) {
		j = i + threadIdx.x;
		E_priv = (A[j*(%(N)s + 1)] + R[j*(%(N)s + 1)]) > 0;
		e[((iteration-1) %% %(CONVITS)s)*%(N)s + j] = E_priv;
		E[j] = E_priv;
		atomicAdd(&K[0], E_priv);
	}
	__syncthreads();

	// Only check if we go past convergence iterations
	if (iteration >= %(CONVITS)s) {
		for (i=0; i<%(N)s; i+=blockDim.x) {
			j = i + threadIdx.x;
			for (k=0; k<%(CONVITS)s; k++)
				e_sum += e[k*%(N)s + j];
			se[j] = e_sum;
			temp += (e_sum == %(CONVITS)s) + (e_sum == 0);
		}
		atomicAdd(&conv_sum[0], temp);
		__syncthreads();
		converged[0] = (conv_sum[0] == %(N)s) && (K[0] > 0);// && (threadIdx.x == 0);
	}
}
"""

# N = 32
# NEG_MAX = np.finfo('float32').min
# CONVITS = 100
# MAXITS = 1000
# DAMPFACT = 0.9
# mod = compiler.SourceModule(kernelCUDA % {'N':N, 'NEG_MAX':NEG_MAX, 'CONVITS':CONVITS, 'DAMP':DAMPFACT})
# similarity = mod.get_function("similarity")
# preference = mod.get_function("preference")
# responsibilities = mod.get_function("responsibilities")
# availabilities = mod.get_function("availabilities")
# convergence = mod.get_function("convergence")
