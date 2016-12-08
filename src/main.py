import kernel as ker
import utility as ut
import numpy as np
import time

# Environment for the algorithm
#------------------------------------------------|
N = 2048
NEG_MAX = np.finfo('float32').min
CONVITS = 100
MAXITS = 1000
DAMPFACT = 0.9
#------------------------------------------------|

start_prog = time.time()
# Read point cloud data
#------------------------------------------------|
x, y, z = ut.readPointCloud('./data/data.xyz', N)
xi = np.copy(x).astype(np.float32)
yi = np.copy(y).astype(np.float32)
zi = np.copy(z).astype(np.float32)
#------------------------------------------------|

# Compiling and getting kernel functions
#------------------------------------------------|
mod = ker.compiler.SourceModule(ker.kernelCUDA % {'N':N, 'NEG_MAX':NEG_MAX, 'CONVITS':CONVITS, 'DAMP':DAMPFACT})
similarity = mod.get_function("similarity")
preference = mod.get_function("preference")
responsibilities = mod.get_function("responsibilities")
availabilities = mod.get_function("availabilities")
convergence = mod.get_function("convergence")
#------------------------------------------------|

start_ker = time.time()
# Similarity Matrix calculation
#------------------------------------------------|
x_gpu = ker.gpuarray.to_gpu(xi)
y_gpu = ker.gpuarray.to_gpu(yi)
z_gpu = ker.gpuarray.to_gpu(zi)
sim_gpu = ker.gpuarray.zeros((N,N), np.float32)
print "len(x): ", N

block_dim = 32
grid_dim = int(np.ceil(N/block_dim))
similarity(x_gpu, y_gpu, z_gpu, sim_gpu,
	grid=(grid_dim, grid_dim, 1),
	block=(block_dim, block_dim,1))
sim_cpu = sim_gpu.get()

# Calculate preference
#-----------------------------------
preference(sim_gpu,
	grid=(1,1,1),
	block=(1024,1,1))
sim_cpu = sim_gpu.get()
#-----------------------------------

# Message passing procedure
#---------------------------------------------
i = 0
A_cpu = np.zeros((N,N), np.float32)
R_cpu = np.zeros((N,N), np.float32)
AS_gpu = ker.gpuarray.zeros((N,N), np.float32)
RP_gpu = ker.gpuarray.zeros((N,N), np.float32)
E_cpu = np.zeros(N, np.bool)
e_gpu = ker.gpuarray.zeros((CONVITS, N), np.uint32)
se_gpu = ker.gpuarray.zeros(N, np.uint32)
converged_gpu = ker.gpuarray.zeros(1, np.bool)
converged_cpu = False

while not converged_cpu and i < MAXITS:
	i += 1

	S_gpu = ker.gpuarray.to_gpu(sim_cpu)
	A_gpu = ker.gpuarray.to_gpu(A_cpu) # A inherently transposed
	R_gpu = ker.gpuarray.to_gpu(R_cpu)
	
	responsibilities(S_gpu, R_gpu, A_gpu, AS_gpu,
		grid=(N,1,1),
		block=(1024,1,1))
	R_cpu = R_gpu.get()

	_A = A_cpu.T.copy()
	_R = R_cpu.T.copy()
	
	A_gpu = ker.gpuarray.to_gpu(_A)
	R_gpu = ker.gpuarray.to_gpu(_R)

	iteration = 0
	availabilities(A_gpu, R_gpu, RP_gpu, np.int32(iteration),
		grid=(N,1,1),
		block=(1024,1,1))
	A_cpu = A_gpu.get().T.copy()

	E_gpu = ker.gpuarray.to_gpu(E_cpu)
	A_gpu = ker.gpuarray.to_gpu(np.diag(A_cpu))
	R_gpu = ker.gpuarray.to_gpu(np.diag(R_cpu))

	convergence(A_gpu, R_gpu, E_gpu, e_gpu, se_gpu, np.int32(i), converged_gpu,
		grid = (1,1,1),
		block = (1024,1,1))
	E_cpu = E_gpu.get()
	converged_cpu = converged_gpu.get()
kernel_time = float(time.time()-start_ker)
print "Number of clusters: ", np.sum(E_cpu)
print "Exemplars:\n", np.argwhere(E_cpu)
program_time = float(time.time()-start_prog)

print "Kernel time: ", kernel_time
print "Program time: ", program_time
