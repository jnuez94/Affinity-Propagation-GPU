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
"""

mod = compiler.SourceModule(kernelCUDA)
similarity = mod.get_function("similarity")
# kernelCL = """

# """
