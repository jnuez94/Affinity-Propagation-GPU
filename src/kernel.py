#!/usr/bin/env python
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernelCUDA = """
__global__ void similarity(float* x, float* y, float* z, float* similarity, int size) {
	
	unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

	float dist = 0;
	
	dist = -(pow((x[row]-x[col]), 2.0) + pow((y[row]-y[col]), 2.0) + pow((z[row]-z[col]), 2.0));

	similarity[row*size+col] = dist;
	__syncthreads();
}
"""
kernelCL = """

"""