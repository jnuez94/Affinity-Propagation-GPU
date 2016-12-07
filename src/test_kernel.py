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

# TEST RESPONSIBILITY KERNEL

j = 0;  #1st row
A = np.zeros((N,N), np.float32)
R = np.zeros((N,N), np.float32)
S_gpu = ker.gpuarray.to_gpu(sim_cpu[j,:])
A_gpu = ker.gpuarray.to_gpu(A[j,:])
R_gpu = ker.gpuarray.to_gpu(R[j,:])
AS_gpu = ker.gpuarray.zeros(R[j,:].shape, np.float32)

#CPU
ss = sim_cpu[j,:] # get all s(i,k)
a_s = A[j,:] + ss # compute a(i,k) + s(i,k)
Y = np.max(a_s).astype(np.float32) # get the max of a(i,k) + s(i,k)
I = np.argmax(a_s)
a_s[I] = np.finfo('float32').min # for r(i,k) where max(a+s) occurs at (i,k), need to find the next maximum that occurs (see eqn) so that the max occurs at (i,k') s.t. k != k'
Y2 = np.max(a_s).astype(np.float32) # find the next max
I2 = np.argmax(a_s)
r = ss - Y # do s(i,k) - max(a+s)
r[I] = ss[I] - Y2 # replace w/ s(i,k) - max(a(i,k')+s(i,k')), k'!=k if max(a+s) was at (i,k)
R[j,:] = (1-DAMPFACT) * r + DAMPFACT * R[j,:]  # dampen

# GPU
ker.responsibilities(S_gpu, R_gpu, A_gpu, AS_gpu,
	grid=(j+1,1,1),
	block=(1024,1,1))
R_cpu = R_gpu.get()
A_cpu = A_gpu.get()
print R_cpu
print A_cpu
print "Validation: ", np.allclose(R, R_cpu)
print "Validation: ", np.allclose(A, A_cpu)
