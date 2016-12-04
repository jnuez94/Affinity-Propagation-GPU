import kernel as ker
import utility as ut
import numpy as np

x, y, z = ut.readPointCloud('../data/short.xyz')
xi = np.copy(x).astype(np.float32)
yi = np.copy(y).astype(np.float32)
zi = np.copy(z).astype(np.float32)

x_gpu = ker.gpuarray.to_gpu(xi)
y_gpu = ker.gpuarray.to_gpu(yi)
z_gpu = ker.gpuarray.to_gpu(zi)
sim_gpu = ker.gpuarray.zeros((len(x),len(x)), np.float32)
print "len(x): ", len(x)
print "shape: ", sim_gpu.shape
print "shape: ", np.copy(sim_gpu.get()).astype(np.float32).shape
sim = ut.pysimilarity(x,y,z)

block_dim = 32
grid_dim = len(x)/block_dim + 1
ker.similarity(x_gpu, y_gpu, z_gpu, sim_gpu, np.int32(32),
	grid=(grid_dim, grid_dim, 1),
	block=(block_dim, block_dim,1))
sim_cpu = sim_gpu.get()
print(sim_cpu)
print "Validation: ", np.allclose(sim, sim_cpu)