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
N = len(x)
sim_gpu = ker.gpuarray.zeros((N,N), np.float32)
print "len(x): ", N
print "shape: ", sim_gpu.shape
print "shape: ", np.copy(sim_gpu.get()).astype(np.float32).shape
sim = ut.pysimilarity(x,y,z)

block_dim = 32
grid_dim = N/block_dim + 1
ker.similarity(x_gpu, y_gpu, z_gpu, sim_gpu,
	grid=(grid_dim, grid_dim, 1),
	block=(block_dim, block_dim,1))
sim_cpu = sim_gpu.get()
print(sim_cpu)
print "Validation: ", np.allclose(sim, sim_cpu)

ker.preference(sim_gpu,
	grid=(1,1,1),
	block=(1024,1,1))
sim_cpu = sim_gpu.get()
print(sim_cpu)
# CPU update preference
pref = np.mean(sim)
for i in range(N):
	sim[i,i] = pref
print "Validation: ", np.allclose(sim, sim_cpu)
