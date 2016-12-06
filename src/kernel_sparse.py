#!/usr/bin/env python
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernelCUDA = """

// 1024 threads per block, N blocks.
// Each block calculates similarities relative to the blockIdx.xth point.
// First block also calculates the preferences
__global__ void similarity(float* x, float* y, float* z, float* S) {
    unsigned int j;
    unsigned int k;
    unsigned float dist;
    unsigned int blk_os = blockIdx.x*(%(N)s-1);

    // Populate similarity table for i != k
    for (k=0; k<%(N)s; k+=blockDim.x) {
        j = k*blockDim.x+threadIdx.x;
        if (j==blockIdx.x) continue;
        if (j>blockIdx.x) j--;
        S[blk_os + j] = blockIdx.x;
        S[%(N)s^2 + blk_os + j] = k+threadIdx.x;
        dist = 0;
        dist -= (x[blockIdx.x] - x[k+threadIdx.x])^2;
        dist -= (y[blockIdx.x] - y[k+threadIdx.x])^2;
        dist -= (z[blockIdx.x] - z[k_threadIdx.x])^2;
        S[2*%(N)s^2 + blk_os + j] = dist;
    }
}

// Calculate the preference with a single block
// Should be median but this is a mean
__global__ void preference(float* S) {
    unsigned int j;
    unsigned int k;
    unsigned int blk_os = %(N)s*%(N)s-%(N)s;
    float P_priv = 0;
    __shared__ float P_sh;

    for (j=0; j<blk_os; j+=blockDim.x)
        P_priv += S[2*%(N)s^2 + j + threadIdx.x]
    atomicAdd(&P_loc, P_priv/blk_os);
    __syncthreads();
    for (j=0; j<%(N)s; j+=blockDim.x) {
        k = blk_os + j + threadIdx.x;
        S[k] = j + threadIdx.x;
        S[%(N)s^2 + k] = j + threadIdx.x;
        S[2*%(N)s^2 + k] = P_loc;
    }
}

// Kernel passes messages and checks for convergence
__global__ void apupdate(float* S, float* R, float* A) {
    unsigned int blk_os;
    unsigned int i;
    unsigned int j;
    unsigned int k;
    float temp = 0; //temp register
    __shared__ float AS[%(N)s];
    __shared__ float atom_temp = -FLOAT_MAX;
    __shared__ int max_idx;
    __shared__ bool dn = false; //convergence flag

    while(!dn) {
        // COMPUTE RESPONSIBILITIES
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
"""

mod = compiler.SourceModule(kernelCUDA % {'N':N, 'CONVITS':CONVITS, 'DAMP':DAMPFACT})
similarity = mod.get_function("similarity")
preference = mod.get_function("preference")
apupdate = mod_get_function("apupdate")
# kernelCL = """

# """
