#!/usr/bin/env python
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernelCUDA = """

__global__ void similarity(float* x, float* y, float* z, float* s) {

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
__global__ void preference(float* S) {
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
/*
// Kernel passes messages. Each block of 1024 threads computes 1 row (N) of A & R.
__global__ void apupdate(float* S, float* R, float* A) {
	unsigned int blk_os;
	unsigned int i;
	unsigned int j;
	unsigned int k;
	float temp = 0; //temp register
	__shared__ float old[%(N)s]; //temp for previous R or A
	__shared__ float AS[%(N)s]; //A+S
	// shared variable that stores our atomic operation results
	// TODO: need to make it a pointer for atomic operations
	__shared__ float atom_temp = -FLOAT_MAX;
	__shared__ int max_idx; //index of maximums
	//__shared__ bool dn = false; //convergence flag

	//while (!dn) {

		// COMPUTE RESPONSIBILITY
		blk_os = blockIdx.x*%(N)s/blockDim.x + threadIdx.x;
		// Reads in old R, adds S & A, finds max of the sum
		for (i=0; i<%(N)s; i+=blockDim.x) {
			j = i + threadIdx.x;
			old[j] = R[blk_os + j];
			float AS_priv = S[blk_os + j] + A[blk_os + j];
			atomicMax(&atom_temp, AS_priv);
			AS[j] = AS_priv;
		}
		__syncthreads();

		// Updates responsibility, sets the max to -infinity
		for (i=0; i<%(N)s; i+=blockDim.x) {
			j = i + threadIdx.x;
			R[blk_os + j] = S[blk_os + j] - atom_temp;
			if (AS[j] == atom_temp) {
				AS[j] = __int_as_float(0xff800000); //-infinity
				max_idx = j;
			}
		}
		__syncthreads();

		// Find new max
		for (i=0; i<%(N)s; i+=blockDim.x) {
			atomicMax(&atom_temp, AS[i + threadIdx.x]);
		}
		__syncthreads();

		// Update responsibility again at the max index only
		if (threadIdx.x == 0)
			R[blk_os + max_idx] = S[blk_os + max_idx] - atom_temp;
		__syncthreads();

		// Apply damping
		for (i=0; i<%(N)s; i+= blockDim.x) {
			j = i + threadIdx.x;
			temp = (1-%(DAMP)s)*R[blk_os + j] + %(DAMP)s*old[j];
			if (temp > FLT_MAX)
				R[blk_os + j] = FLT_MAX;
			else
				R[blk_os + j] = temp;
		}
		__syncthreads();

		// COMPUTE AVAILABILITY
		// Note: indexing is transposed

		// Read in old A, find elementwise max of R, 0
		j = threadIdx.x*%(N)s;
		for (i=0; i<%(N)s; i+=blockDim.x) {
			blk_os = i*%(N)s + blockIdx.x;
			k = i + threadIdx.x;
			old[k] = A[blk_os + j];
			if (R[blk_os + j] > 0) //elementwise max
				Rp[k] = R[blk_os + j];
			else
				Rp[k] = 0;
			atomicAdd(&atom_temp, Rp[k]); //sum Rp
		}
		__syncthreads();

		// Update availability
		for (i=0; i<%(N)s; i+=blockDim.x) {
			blk_os = i*%(N)s + blockIdx.x;
			k = i + threadIdx.x;
			A[blk_os + j] = atom_temp - Rp[k];
		}
		__syncthreads();

		// Grab diagonal
		if (threadIdx.x == 0)
			temp = A[blk_Idx.x*%(N)s + blockIdx.x];
		__syncthreads();

		// Set A to elementwise min of A, 0
		for (i=0; i<%(N)s; i+=blockDim.x) {
			blk_os = i*%(N)s + blockIdx.x;
			if (A[blk_os + j] > 0)
				A[blk_os + j] = 0;
		}
		__syncthreads();

		// Replace diagonal
		if (threadIdx.x == 0)
			A[blockIdx.x*%(N)s + blockIdx.x] = dA;
		__syncthreads();

		// Apply damping
		for (i=0; i<%(N)s; i+= blockDim.x) {
			blk_os = i*%(N)s + blockIdx.x;
			k = i + threadIdx.x;
			temp = (1-%(DAMP)s)*A[blk_os + j] + %(DAMP)s*old[k];
			A[blk_os + j] = temp;
		}
		__syncthreads();

		// DONE - NEXT KERNEL CHECKS FOR CONVERGENCE
		// SHIT - WE NEED TO SUM E = (diag(A) + diag(R)) > 0
		// kernel is called again if unconverged
}

// Check for convergence on the diagonal of A & R
// Use 1 block of 1024 threads
// Note the converged flag only tells CPU to stop iterating.
// CPU will not call this if MAXITS hit, so maxits checks are removed
__global__ void convergence(float* A, float* R, int* E, bool *e, int iteration, bool converged) {
	unsigned int i;
	unsigned int j;
	unsigned int k;
	unsigned int temp = 0;
	float RA;
	__shared__ bool E[%(N)s];
	__shared__ int K; //sum of E
	__shared__ int se[%(N)s];
	__shared__ int conv_sum;

	// E is vector of whether diagonal (A+R) > 0
	for (i=0; i<%(N)s; i+=blockDim.x) {
		j = i + threadIdx.x;
		RA = A[j*%(N)s + j] + R[j*%(N)s + j];
		if (RA > 0) E[j] = true;
		else E[j] = false;
		e[((iteration-1) % %(CONVITS)s)*%(N)s + j] = E[j];
		atomicAdd(&K, E[j]);
	}
	__syncthreads();

	// Only check if we go past convergence iterations
	if (iteration >= %(CONVITS)s) {
		for (i=0; i<%(N)s; i+=blockDim.x) {
			j = i + threadIdx.x;
			for (k=0; k<%(CONVITS)s; k++)
				temp += e[k*%(N)s + j];
			se[j] = temp;
			temp = 0;
			if (se[j] == %(CONVITS)s) temp++;
			if (se[j] == 0) temp++;
			atomicAdd(&conv_sum, temp);
		}
		__syncthreads();
		if (conv_sum == N && K > 0 && threadIdx.x == 0)
			converged = true;
	}
}
*/
"""

N = 32
CONVITS = 100
MAXITS = 1000
DAMPFACT = 0.9
mod = compiler.SourceModule(kernelCUDA % {'N':N, 'CONVITS':CONVITS, 'DAMP':DAMPFACT})
similarity = mod.get_function("similarity")
preference = mod.get_function("preference")
#apupdate = mod.get_function("apupdate")
#convergence = mod.get_function("convergence")
