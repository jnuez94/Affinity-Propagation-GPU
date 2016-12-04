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

__global__ void apupdate(float* similarity, float* responsibility, float* availability, 
	int size) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx >= size)
		return;

	// Message passing
	bool dn = false;
	unsigned int i = 0;
	float old = 0.0;
	float AS = 0.0;

	while (!dn) {
		i += 1;

		for (k = 0; k < size; ++k) {
			old = resp[row*size+col];

		}
	}

}
"""

mod = compiler.SourceModule(kernelCUDA)
similarity = mod.get_function("similarity")
# kernelCL = """

# """