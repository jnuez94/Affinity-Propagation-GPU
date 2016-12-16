import numpy as np
import utility as ut
import time
np.set_printoptions(threshold = np.inf) # To print the full array

"""
Python version of apclusterSparse.m
utility.py reads data and generates float32 numpy array of coordinates
Square similarity matrix is generated here
"""
# PARAMETERS
N = 1024
MAXITS = 1000 # maximum iterations. Default = 1000
CONVITS = 100 # converged if est. centers stay fixed for convits iterations. Default = 100
DAMPFACT = 0.9 # 0.5 to 1, damping. Higher needed if oscillations occur. Default = 0.9.
PLT = True # Plots net similarity after each iteration
if DAMPFACT>0.9:
    print 'Large damping factor, turn on plotting! Consider using larger value of convits.'

if PLT: netsim = np.zeros(MAXITS+1, np.float32)

# START TIMER
start_prg = time.time()

# Read point cloud
data = np.array(ut.readPointCloud('../data/data.xyz', N))
# Generate similarity matrix
start_ker = time.time()
start_sim = time.time()
s = np.zeros((N,N), np.float32)
for i in range(N):
    for j in range(N):
        s[i,j] = -np.sum(np.square(data[:,i] - data[:,j]))
sim_time = float(time.time()-start_sim)
# Set preference to median similarity
start_pref = time.time()
p = np.mean(s).astype(np.float32)
# Update diagonal of sparse matrix with preference
for i in range(N):
    s[i,i] = p
# Make vector of preferences
p = p*np.ones(N, np.float32)
pref_time = float(time.time()-start_pref)

# Allocate space for messages
A = np.zeros((N,N), np.float32)
R = np.zeros((N,N), np.float32)
E = np.zeros(N, np.int32)

# Execute parallel affinity propagation updates
e = np.zeros((CONVITS, N), np.int32)
dn = 0
i = 0
# Initialize timers
resp_times = []
avail_times = []
conv_times = []
while not dn:
    i += 1

    # Compute responsibilities
    start_resp = time.time()
    for j in range(N): #looping over i
        ss = s[j,:] # get all s(i,k)
        a_s = A[j,:] + ss # compute a(i,k) + s(i,k)
        Y = np.max(a_s).astype(np.float32) # get the max of a(i,k) + s(i,k)
        I = np.argmax(a_s)
        a_s[I] = np.finfo('float32').min # for r(i,k) where max(a+s) occurs at (i,k), need to find the next maximum that occurs (see eqn) so that the max occurs at (i,k') s.t. k != k'
        Y2 = np.max(a_s).astype(np.float32) # find the next max
        I2 = np.argmax(a_s)
        r = ss - Y # do s(i,k) - max(a+s)
        r[I] = ss[I] - Y2 # replace w/ s(i,k) - max(a(i,k')+s(i,k')), k'!=k if max(a+s) was at (i,k)
        R[j,:] = (1-DAMPFACT) * r + DAMPFACT * R[j,:]  # dampen
    resp_times.append(float(time.time()-start_resp))

    # Compute availabilities
    start_avail = time.time()
    for j in range(N): #looping over k
        rp = np.maximum(R[:,j], 0) # elementwise maximum of r transposed
        rp[j] = R[j,j] # replace r(k,k) which is not subject to max
        a = np.sum(rp) - rp # a(k,k) = sum(max{0,r(i',k)}) s.t. i'!=k, else = sum(max{0,r(i',k)}) + r(k,k) - r(i,k), i'!=k which is equivalent to sum(max{0,r(i'k)}) + r(k,k) for i'!=i,k
        dA = a[j] # grab a(k,k) which is not subject to min
        a = np.minimum(a, 0) # elementwise minimum for a(i,k)
        a[j] = dA # replace a(k,k)
        A[:,j] = (1-DAMPFACT) * a + DAMPFACT * A[:,j] # dampen
    avail_times.append(float(time.time()-start_avail))

    # Check for convergence
    start_conv = time.time()
    for j in range(N):
        E[j] = (A[j,j] + R[j,j]) > 0 # Find where A(i,i)+R(i,i) is > 0 (i.e. find the exemplars)
    e[(i-1) % CONVITS , :] = E # Buffer for convergence iterations
    K = np.sum(E).astype(np.int32) # How many exemplars are there?
    if i >= CONVITS or i>= MAXITS:
        se = np.sum(e, 0).astype(np.int32) # Sum all convergence iterations
        unconverged = np.sum((se==CONVITS) + (se==0)) != N # Unconverged if # of exemplars isn't same for CONVITS
        if (not unconverged and K>0) or (i==MAXITS): # Stop the message passing loop
            dn=1
    conv_times.append(float(time.time()-start_conv))

kernel_time = float(time.time()-start_ker)
resp_time = np.mean(resp_times)
avail_time = np.mean(avail_times)
conv_time = np.mean(conv_times)
print 'Cluster indices:', np.argwhere(E).flatten()

# Identify exemplars
for j in range(N):
    E[j] = (A[j,j] + R[j,j]) > 0
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

program_time = float(time.time() - start_prg)

if PLT:
    print '\nNumber of identified clusters: %d\n' % K
    print 'Fitness (net similarity): %f\n' % tmpnetsim
    print '  Similarities of data points to exemplars: %f\n' % dpsim
    print '  Preferences of selected exemplars: %f\n' % tmpexpref
    print 'Number of iterations: %d\n' % i
    print 'Time taken for entire Python program: %f\n' % program_time
    print 'Time taken for parallelized portion: %f\n' % kernel_time
    print 'Average time of responsibility update: %f\n' % resp_time
    print 'Average time of availability update: %f\n' % avail_time
    print 'Average time of convergence check: %f\n\n' % conv_time
if unconverged:
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
        h = fig.scatter(data[0,ii], data[1,ii], zs=data[2,ii])
        plt.hold(True)
        col = np.tile(np.random.rand(1,3), (np.size(ii), 1))
        plt.setp(h, color=col, facecolor=col)
        for j in ii:
            fig.plot(np.hstack((data[0,j], data[0,int(i)])),
                     np.hstack((data[1,j], data[1,int(i)])),
                     zs=np.hstack((data[2,j], data[2,int(i)])),
                     color=col[0,:])
        fig.set_xlabel('x')
        fig.set_ylabel('y')
        fig.set_zlabel('z')
        plt.draw()
    plt.axis('image')
    plt.show(block=True)
    # Grid and 3D rotation w/ mouse enabled by default
