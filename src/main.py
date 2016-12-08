import kernel as ker
import utility as ut
import numpy as np

# Read point cloud data
#------------------------------------------------|
x, y, z = ut.readPointCloud('./data/short.xyz')
xi = np.copy(x).astype(np.float32)
yi = np.copy(y).astype(np.float32)
zi = np.copy(z).astype(np.float32)
#------------------------------------------------|

# Environment for the algorithm
#------------------------------------------------|
N = len(x)
NEG_MAX = np.finfo('float32').min
CONVITS = 100
MAXITS = 1000
DAMPFACT = 0.9
#------------------------------------------------|

# Compiling and getting kernel functions
#------------------------------------------------|
mod = compiler.SourceModule(kernelCUDA % {'N':N, 'NEG_MAX':NEG_MAX, 'CONVITS':CONVITS, 'DAMP':DAMPFACT})
similarity = mod.get_function("similarity")
preference = mod.get_function("preference")
responsibilities = mod.get_function("responsibilities")
availabilities = mod.get_function("availabilities")
convergence = mod.get_function("convergence")
#------------------------------------------------|

# Similarity Matrix calculation
#------------------------------------------------|
x_gpu = ker.gpuarray.to_gpu(xi)
y_gpu = ker.gpuarray.to_gpu(yi)
z_gpu = ker.gpuarray.to_gpu(zi)
sim_gpu = ker.gpuarray.zeros((N,N), np.float32)
print "len(x): ", N

grid_dim = N/block_dim + 1
ker.similarity(x_gpu, y_gpu, z_gpu, sim_gpu,
	grid=(grid_dim, grid_dim, 1),
	block=(N, N,1))
sim_cpu = sim_gpu.get()
# print(sim_cpu)
print "S Validation: ", np.allclose(sim, sim_cpu)