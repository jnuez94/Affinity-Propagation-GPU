import kernel as ker
import utility as ut
import numpy as np

x, y, z = ut.readPointCloud('./data/short.xyz')
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

# CONSTANTS
#j = 0;  #1st row
DAMPFACT = 0.9
CONVITS = 100
MAXITS = 1000


dn = 0
i = 0

A = np.zeros((N,N), np.float32)
R = np.zeros((N,N), np.float32)
A_cpu = np.zeros((N,N), np.float32)
R_cpu = np.zeros((N,N), np.float32)
AS_gpu = ker.gpuarray.zeros((N,N), np.float32)
RP = np.zeros((N,N), np.float32)
RP_gpu = ker.gpuarray.zeros((N,N), np.float32)
E = np.zeros(N, np.bool)
E_cpu = np.zeros(N, np.bool)
e_gpu = ker.gpuarray.zeros((CONVITS, N), np.int32)
se_gpu = ker.gpuarray.zeros(N, np.int32)
converged_gpu = ker.gpuarray.zeros(1, np.bool)

while not dn and i < MAXITS:
	i += 1

	# TEST RESPONSIBILITY KERNEL
	S_gpu = ker.gpuarray.to_gpu(sim_cpu)
	A_gpu = ker.gpuarray.to_gpu(A_cpu)
	R_gpu = ker.gpuarray.to_gpu(R_cpu)

	#CPU
	for j in range(N):
		ss = sim_cpu[j,:] # get all s(i,k)
		a_s = A[j,:] + ss # compute a(i,k) + s(i,k)
		Y = np.max(a_s).astype(np.float32) # get the max of a(i,k) + s(i,k)
		I = np.argmax(a_s)
		a_s[I] = np.finfo('float32').min # for r(i,k) where max(a+s) occurs at (i,k), need to find the next maximum 	that 	occurs (see eqn) so that the max occurs at (i,k') s.t. k != k'
		Y2 = np.max(a_s).astype(np.float32) # find the next max
		I2 = np.argmax(a_s)
		r = ss - Y # do s(i,k) - max(a+s)
		r[I] = ss[I] - Y2 # replace w/ s(i,k) - max(a(i,k')+s(i,k')), k'!=k if max(a+s) was at (i,k)
		R[j,:] = (1-DAMPFACT) * r + DAMPFACT * R[j,:]  # dampen

	# GPU
	ker.responsibilities(S_gpu, R_gpu, A_gpu, AS_gpu,
		grid=(N,1,1),
		block=(N,1,1))
	R_cpu = R_gpu.get()
	A_cpu = A_gpu.get()
	#print R_cpu
	#print A_cpu
	print "R Validation: ", np.allclose(R, R_cpu)
	#print "A Validation: ", np.allclose(A, A_cpu)
	#print R[j,:]
	#print A[j,:]

	# TEST AVAILABILITY KERNEL
	A_gpu = ker.gpuarray.to_gpu(A_cpu.T.copy())
	R_gpu = ker.gpuarray.to_gpu(R_cpu.T.copy())

	#CPU
	for j in range(N):
		rp = np.maximum(R[:,j], 0) # elementwise maximum of r transposed
		rp[j] = R[j,j] # replace r(k,k) which is not subject to max
		a = np.sum(rp) - rp # a(k,k) = sum(max{0,r(i',k)}) s.t. i'!=k, else = sum(max{0,r(i',k)}) + r(k,k) - r(i,k), i'!=k 	which is equivalent to sum(max{0,r(i'k)}) + r(k,k) for i'!=i,k
		dA = a[j] # grab a(k,k) which is not subject to min
		a = np.minimum(a, 0) # elementwise minimum for a(i,k)
		a[j] = dA # replace a(k,k)
		A[:,j] = (1-DAMPFACT) * a + DAMPFACT * A[:,j] # dampen

	#GPU
	iteration = 0
	ker.availabilities(A_gpu, R_gpu, RP_gpu, np.int32(iteration),
		grid=(N,1,1),
		block=(N,1,1))
	A_cpu = A_gpu.get()
	#print A_cpu
	#print A
	print "A Validation: ", np.allclose(A, A_cpu)

	# TEST CONVERGENCE KERNEL
	E_gpu = ker.gpuarray.to_gpu(E_cpu)
	A_gpu = ker.gpuarray.to_gpu(np.diag(A_cpu))
	R_gpu = ker.gpuarray.to_gpu(np.diag(R_cpu))

	#CPU
	for j in range(N):
		E[j] = (A[j,j] + R[j,j]) > 0 # Find where A(i,i)+R(i,i) is > 0 (i.e. find the exemplars)
	e[(i-1) % CONVITS , :] = E # Buffer for convergence iterations
	K = np.sum(E).astype(np.int32) # How many exemplars are there?
	if i >= CONVITS or i>= MAXITS:
		se = np.sum(e, 0).astype(np.int32) # Sum all convergence iterations
		unconverged = np.sum((se==CONVITS) + (se==0)) != N # Unconverged if # of exemplars isn't same for CONVITS
		if (not unconverged and K>0) or (i==MAXITS): # Stop the message passing loop
		dn=1

	#GPU
	ker.convergence(A_gpu, R_gpu, E_gpu, e_gpu, se_gpu, np.int32(i), converged_gpu,
		grid = (1,1,1),
		block = (N,1,1))
	E_cpu = E_gpu.get()
	converged_cpu = converged_gpu.get()

print E_cpu
print E
print "E Validation", np.allclose(E, E_cpu)
print dn, converged_cpu
