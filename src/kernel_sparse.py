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
__global__ void apupdate
"""

mod = compiler.SourceModule(kernelCUDA % {'N':N, 'CONVITS':CONVITS, 'DAMP':DAMPFACT})
similarity = mod.get_function("similarity")
preference = mod.get_function("preference")
apupdate = mod_get_function("apupdate")
# kernelCL = """

# """
