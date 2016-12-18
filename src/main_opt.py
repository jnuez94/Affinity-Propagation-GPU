import kernel_opt as ker
import utility as ut
import numpy as np
import time

# Environment for the algorithm
#------------------------------------------------|
N = 1024
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
S_gpu = ker.gpuarray.zeros((N,N), np.float32)
print "len(x): ", N

block_dim = 32
grid_dim = int(np.ceil(N/block_dim))

start_similarity = time.time() 										# Similarity timing
similarity(x_gpu, y_gpu, z_gpu, S_gpu,
	grid=(grid_dim, grid_dim, 1),
	block=(block_dim, block_dim,1))
similarity_time = float(time.time()-start_similarity) 		# End of Sim timing

# Calculate preference
#-----------------------------------
start_pref = time.time()
preference(S_gpu,
	grid=(1,1,1),
	block=(1024,1,1))
pref_time = float(time.time()-start_pref)
#-----------------------------------

# Message passing procedure
#---------------------------------------------
i = 0
A_gpu = ker.gpuarray.zeros((N,N), np.float32)
R_gpu = ker.gpuarray.zeros((N,N), np.float32)
AS_gpu = ker.gpuarray.zeros((N,N), np.float32)
RP_gpu = ker.gpuarray.zeros((N,N), np.float32)
E_gpu = ker.gpuarray.zeros(N, np.bool)
e_gpu = ker.gpuarray.zeros((CONVITS, N), np.uint32)
se_gpu = ker.gpuarray.zeros(N, np.uint32)
converged_gpu = ker.gpuarray.zeros(1, np.bool)
converged_cpu = False

resp_time = []
avail_time = []
conv_time = []

start_msgpassing = time.time()
while not converged_cpu and i < MAXITS:
	i += 1

	start_resp = time.time()
	responsibilities(S_gpu, R_gpu, A_gpu, AS_gpu,
		grid=(N,1,1),
		block=(1024,1,1))
	resp_time.append(float(time.time()-start_resp))

	start_avail = time.time()
	availabilities(A_gpu, R_gpu, RP_gpu,
		grid=(N,1,1),
		block=(1024,1,1))
	avail_time.append(float(time.time()-start_avail))

	start_conv = time.time()
	convergence(A_gpu, R_gpu, E_gpu, e_gpu, se_gpu, np.int32(i), converged_gpu,
		grid = (1,1,1),
		block = (1024,1,1))
	conv_time.append(float(time.time()-start_conv))

	converged_cpu = converged_gpu.get()

E = E_gpu.get()

msgpassing_time = float(time.time()-start_msgpassing)
kernel_time = float(time.time()-start_ker)
print "Number of clusters: ", np.sum(E)
print "Exemplars:\n", np.argwhere(E)

s = sim_cpu
p = s[0,0]*np.ones(N, np.float32)
# Identify exemplars
# for j in range(N):
#     E[j] = (A[j,j] + R[j,j]) > 0
K = np.sum(E).astype(np.int32)
if K>0:
    tmpidx=np.zeros(N, np.float32)
    tmpidx[np.argwhere(E)] = np.argwhere(E) # store index of exemplar as itself
    for j in np.argwhere(E==0).flatten(): # for non-exemplars
        ss = s[j,:] # get similarity for non-exemplars
        ii = np.arange(N).astype(np.int32) # 0 to N-1
        ee = np.argwhere(E[ii]) # indices of exemplars
        #smx = np.max(ss[ee])
        imx = np.argmax(ss[ee]) # find the exemplar that jth point belongs to
        tmpidx[j] = ii[ee[imx]] # store the index of that exemplar
    EE = np.zeros(N, np.float32)
    for j in np.argwhere(E).flatten(): # for exemplars
        jj = np.argwhere(tmpidx==j).flatten() # jj contains all points in a cluster
        mx = float('-Inf')
        ns = np.zeros(N, np.float32)
        msk = np.zeros(N, np.float32)
        for m in jj: # for all points in cluster
            mm = np.arange(N).astype(np.int32) # 0 to N-1
            msk[mm] += 1 # Equals number of points in cluster
            ns[mm] += s[m,:] # Net similarity to each point in the cluster
        ii = jj[np.argwhere(msk[jj] == np.size(jj))] # ii equals the cluster (=jj)
        #smx = np.max(ns[ii])
        imx = np.argmax(ns[ii]) # find max net similarity in cluster
        EE[ii[imx]] = 1 # Set EE at point with max net similarity to 1
    E = EE # Make E contain the few new candidate exemplars
    tmpidx = np.zeros(N, np.float32)
    tmpdpsim = 0
    tmpidx[np.argwhere(E)] = np.argwhere(E) # store indices of the new exemplars
    tmpexpref = np.sum(p[np.argwhere(E)]) # sum of preferences at exemplars (= preference * # of exemplars)
    for j in np.argwhere(E==0).flatten(): # for non-exemplars
        ss = s[j,:] # get similarities
        ii = np.arange(N).astype(np.int32)
        ee = np.argwhere(E[ii]) # indices of exemplars
        smx = np.max(ss[ee]) # find greatest similarity to current point
        imx = np.argmax(ss[ee]) # get index of that point
        tmpidx[j] = ii[ee[imx]] # store the index of that point
        tmpdpsim += smx # sum of max similarities to non-exemplars
    tmpnetsim = tmpdpsim + tmpexpref # net similarity is max similarities + sum of preferences
else:
    tmpidx = float('nan')*np.ones(N)
    tmpnetsim = float('nan')
    tmpexpref = float('nan')

netsim = tmpnetsim
dpsim = tmpnetsim - tmpexpref
expref = tmpexpref
idx = tmpidx

program_time = float(time.time() - start_prog)
PLT = True


if PLT:
    print '\nNumber of identified clusters: %d\n' % K
    print 'Fitness (net similarity): %f\n' % tmpnetsim
    print '  Similarities of data points to exemplars: %f\n' % dpsim
    print '  Preferences of selected exemplars: %f\n' % tmpexpref
    print 'Number of iterations: %d\n' % i
    print 'Time taken for entire Python program: %f\n' % program_time
    print 'Time taken for parallelized portion: %f\n\n' % kernel_time
    print 'Time taken for similarity kernel: %f\n\n' % similarity_time
    print 'Time taken for preference kernel: %f\n\n' % pref_time
    print 'Time taken for message passing: %f\n\n' % msgpassing_time
    print 'Average time taken for responsibility kernel: %f\n\n' % np.mean(resp_time)
    print 'Average time taken for availability kernel: %f\n\n' % np.mean(avail_time)
    print 'Average time taken for convergence kernel: %f\n\n' % np.mean(conv_time)
if not converged_cpu:
    print '\n*** Warning: Algorithm did not converge. The similarities\n'
    print '    may contain degeneracies - add noise to the similarities\n'
    print '    to remove degeneracies. To monitor the net similarity,\n'
    print '    activate plotting. Also, consider increasing maxits and\n'
    print '    if necessary dampfact.\n\n'

# Plot figure showing data and the clusters to compare with Matlab
if PLT:
    import matplotlib as mpl
    #For tesseract server:
    #mpl.use('agg')
    #For Bash on Windows & XLaunch:
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure().gca(projection='3d')
    for i in np.unique(idx):
        ii = np.argwhere(idx == i)
        h = fig.scatter(xi[ii], yi[ii], zs=zi[ii])
        plt.hold(True)
        col = np.tile(np.random.rand(1,3), (np.size(ii), 1))
        plt.setp(h, color=col, facecolor=col)
        for j in ii:
            fig.plot(np.hstack((xi[j], xi[int(i)])),
                     np.hstack((yi[j], yi[int(i)])),
                     zs=np.hstack((zi[j], zi[int(i)])),
                     color=col[0,:])
        fig.set_xlabel('x')
        fig.set_ylabel('y')
        fig.set_zlabel('z')
        plt.draw()
    plt.axis('image')
    plt.show(block=True)
    # Grid and 3D rotation w/ mouse enabled by default
