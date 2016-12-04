#!/usr/bin/env python
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernelCUDA = """

__global__ void similarity(float* x, float* y, float* z, float* similarity, int size) {

	unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

	if (col >= size || row >= size)
		return;

	float dist = 0;
	float xi = x[row]-x[col];
	float yi = y[row]-y[col];
	float zi = z[row]-z[col];

	dist = -(xi*xi + yi*yi + zi*zi);

	similarity[row*size+col] = dist;
	__syncthreads();
}

__global__ void apupdate(float* S, float* R, float* A) {
	unsigned int blk_os;
	unsigned int i;
	unsigned int j;
	__shared__ float old[%(N)s];
	__shared__ float AS[%(N)s];
	__shared__ float max = -FLOAT_MAX;
	__shared__ int max_idx;
	__shared__ bool dn = false;

	// TODO: may need to make max a pointer

	while (!dn) {

		// COMPUTE RESPONSIBILITY
		blk_os = blockIdx.x*%(N)s/blockDim.x + threadIdx.x;
		// Reads in old R, adds S & A, finds max of the sum
		for (i=0; i<%(N)s/blockDim.x; i++) {
			j = i*blockDim.x + threadIdx.x;
			old[j] = R[blk_os + j];
			float AS_priv = S[blk_os + j] + A[blk_os + j];
			atomicMax(&max, AS_priv);
			AS[j] = AS_priv;
		}
		__syncthreads();

		// Updates responsibility, sets the max to -infinity
		for (i=0; i<%(N)s/blockDim.x; i++) {
			j = i*blockDim.x + threadIdx.x;
			R[blk_os + j] = S[blk_os + j] - max;
			if (AS[j] == max) {
				AS[j] = __int_as_float(0xff800000); //-infinity
				max_idx = j;
			}
		}
		__syncthreads();

		// Find new max
		for (i=0; i<%(N)s/blockDim.x; i++) {
			j = i*blockDim.x + threadIdx.x;
			atomicMax(&max, AS[j]);
		}
		__syncthreads();

		// Update responsibility again at the max index only
		if (threadIdx.x == 0)
			R[blk_os + max_idx] = S[blk_os + max_idx] - max;
		__syncthreads();

		// Apply damping
		for (i=0; i<%(N)s/blockDim.x; i++) {
			j = i*blockDim.x + threadIdx.x;
			float temp = (1-%(DAMP)s)*R[blk_os + j] + %(DAMP)s*R[j];
			if (temp > FLT_MAX)
				R[blk_os + j] = FLT_MAX;
			else
				R[blk_os + j] = temp;
		}
		__syncthreads();

		// COMPUTE AVAILABILITY
		// Note: indexing is transposed
		// Read in old A, find max of responsibility
		for (i=0; i<%(N)s/blockDim.x; i++) {
			blk_os = i*%(N)s*blockDim.x + blockIdx.x;
			j = threadIdx.x*%(N)s;
			old[i*blockDim.x+threadIdx.x] = A[blk_os + j];
			atomicMax(&max, R[blk_os + j]);
}
"""

mod = compiler.SourceModule(kernelCUDA)
similarity = mod.get_function("similarity")
# kernelCL = """

# """
