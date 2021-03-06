import numpy as np
import utility as ut
np.set_printoptions(threshold = np.inf) # To print the full array

"""
Python version of apclusterSparse.m
utility.py reads data and generates float32 numpy array of coordinates
Sparse similarity matrix is generated here
"""
# Read point cloud
data = np.array(ut.readPointCloud('../data/short.xyz'))
if np.size(data, 0) != 3: print 'Data should be a 3D point cloud'
if data.dtype != np.float32: print 'Data needs to be float32'
# Generate similarity matrix
N = data.shape[1]
M = N*N
s = np.zeros((N,N), np.float32)
for i in range(N):
    for j in range(N):
        s[i,j] = -np.sum(np.square(data[:,i] - data[:,j]))
# Set preference to median similarity
p = np.mean(s).astype(np.float32)
# Update diagonal of sparse matrix with preference
for i in range(N):
    s[i,i] = p

#PARAMETERS
#defaults for now
MAXITS = 1000 # maximum iterations. Default = 1000
CONVITS = 100 # converged if est. centers stay fixed for convits iterations. Default = 100
DAMPFACT = 0.9 # 0.5 to 1, damping. Higher needed if oscillations occur. Default = 0.9.
PLT = True # Plots net similarity after each iteration
DETAILS = False # Store idx, netsim, dpsim, expref after each iteration
NONOISE = False # Degenerate input similarities with random noise.
if DAMPFACT>0.9:
    print 'Large damping factor, turn on plotting! Consider using larger value of convits.'

########################
# BEGIN MEAT OF FUNCTION
########################

# Make vector of preferences
if np.size(p) == 1: p = p*np.ones(N, np.float32)

"""
# Add a small amount of noise to input similarities
if not NONOISE:
    np.random.seed()
    s[2,0:M] += np.multiply(np.finfo('float32').eps * s[2,0:M] + np.finfo('float32').tiny * 100 , np.random.rand(M))
"""

# Allocate space for messages
A = np.zeros((N,N), np.float32)
R = np.zeros((N,N), np.float32)
E = np.zeros(N, np.int32)
#t = 1
if PLT: netsim = np.zeros(MAXITS+1, np.float32)
if DETAILS:
    idx = np.zeros((MAXITS+1, N), np.float32)
    netsim = np.zeros(MAXITS+1, np.float32)
    dpsim = np.zeros(MAXITS+1, np.float32)
    expref = np.zeros(MAXITS+1, np.float32)

# Execute parallel affinity propagation updates
e = np.zeros((CONVITS, N), np.int32)
dn = 0
i = 0
while not dn:
    i += 1

    # Compute responsibilities
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

    # Compute availabilities
    for j in range(N): #looping over k
        rp = np.maximum(R[:,j], 0) # elementwise maximum of r transposed
        rp[j] = R[j,j] # replace r(k,k) which is not subject to max
        a = np.sum(rp) - rp # a(k,k) = sum(max{0,r(i',k)}) s.t. i'!=k, else = sum(max{0,r(i',k)}) + r(k,k) - r(i,k), i'!=k which is equivalent to sum(max{0,r(i'k)}) + r(k,k) for i'!=i,k
        dA = a[j] # grab a(k,k) which is not subject to min
        a = np.minimum(a, 0) # elementwise minimum for a(i,k)
        a[j] = dA # replace a(k,k)
        A[:,j] = (1-DAMPFACT) * a + DAMPFACT * A[:,j] # dampen

    # Check for convergence
    for j in range(N):
        E[j] = (A[j,j] + R[j,j]) > 0 # Find where A(i,i)+R(i,i) is > 0 (i.e. find the exemplars)
    e[(i-1) % CONVITS , :] = E # Buffer for convergence iterations
    K = np.sum(E).astype(np.int32) # How many exemplars are there?
    if i >= CONVITS or i>= MAXITS:
        se = np.sum(e, 0).astype(np.int32) # Sum all convergence iterations
        unconverged = np.sum((se==CONVITS) + (se==0)) != N # Unconverged if # of exemplars isn't same for CONVITS
        if (not unconverged and K>0) or (i==MAXITS): # Stop the message passing loop
            dn=1

    # Handle plotting and storage of details, if requested
    if PLT or DETAILS:
        if K==0:
            tmpnetsim = float('nan')
            tmpdpsim = float('nan')
            tmpexpref = float('nan')
            tmpidx = float('nan')
        else:
            tmpidx = np.zeros(N)
            tmpdpsim = 0
            tmpidx[np.argwhere(E)] = np.argwhere(E)
            tmpexpref = np.sum(p[np.argwhere(E)])
            discon = 0
            for j in np.argwhere(E==0).flatten():
                ss = s[j,:]
                ii = np.arange(N).astype(np.int32)
                ee = np.argwhere(E[ii])
                if np.size(ee) == 0:
                    discon = 1
                else:
                    smx = np.max(ss[ee])
                    imx = np.argmax(ss[ee])
                    tmpidx[j] = ii[ee[imx]]
                    tmpdpsim += smx
            if discon:
                tmpnetsim = float('nan')
                tmpdpsim = float('nan')
                tmpexpref = float('nan')
                tmpidx = float('nan')
            else:
                tmpnetsim = tmpdpsim + tmpexpref
    if DETAILS:
        netsim[i-1] = tmpnetsim
        dpsim[i-1] = tmpdpsim
        expref[i-1] = tmpexpref
        idx[i-1,:] = tmpidx
    if PLT:
        netsim[i-1] = tmpnetsim
        import matplotlib as mpl
        #For tesseract server:
        #mpl.use('agg')
        #For Bash on Windows & XLaunch:
        mpl.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.figure(1)
        tmp = np.arange(i-1)
        tmpi = np.argwhere(netsim[0:i-1] != float('nan'))
        plt.plot(tmp[tmpi], netsim[tmpi], 'r-')
        plt.xlabel('# Iterations')
        plt.ylabel('Net similarity of quantized intermediate solution')
        plt.draw()
if PLT:
    plt.ion()
    plt.show()

# Identify exemplars
for j in range(N):
    E[j] = (A[j,j] + R[j,j]) > 0
K = np.sum(E).astype(np.int32)
if K>0:
    tmpidx=np.zeros(N, np.float32)
    tmpidx[np.argwhere(E)] = np.argwhere(E)
    for j in np.argwhere(E==0).flatten():
        ss = s[j,:]
        ii = np.arange(N).astype(np.int32)
        ee = np.argwhere(E[ii])
        #smx = np.max(ss[ee])
        imx = np.argmax(ss[ee])
        tmpidx[j] = ii[ee[imx]]
    EE = np.zeros(N, np.float32)
    for j in np.argwhere(E).flatten():
        jj = np.argwhere(tmpidx==j).flatten()
        mx = float('-Inf')
        ns = np.zeros(N, np.float32)
        msk = np.zeros(N, np.float32)
        for m in jj:
            mm = np.arange(N).astype(np.int32)
            msk[mm] += 1
            ns[mm] += s[m,:]
        ii = jj[np.argwhere(msk[jj] == np.size(jj))]
        #smx = np.max(ns[ii])
        imx = np.argmax(ns[ii])
        EE[ii[imx]] = 1
    E = EE
    tmpidx = np.zeros(N, np.float32)
    tmpdpsim = 0
    tmpidx[np.argwhere(E)] = np.argwhere(E)
    tmpexpref = np.sum(p[np.argwhere(E)])
    for j in np.argwhere(E==0).flatten():
        ss = s[j,:]
        ii = np.arange(N).astype(np.int32)
        ee = np.argwhere(E[ii])
        smx = np.max(ss[ee])
        imx = np.argmax(ss[ee])
        tmpidx[j] = ii[ee[imx]]
        tmpdpsim += smx
    tmpnetsim = tmpdpsim + tmpexpref
else:
    tmpidx = float('nan')*np.ones(N)
    tmpnetsim = float('nan')
    tmpexpref = float('nan')
if DETAILS:
    netsim[i] = tmpnetsim
    netsim = netsim[0:i]
    dpsim[i] = tmpnetsim-tmpexpref
    dpsim = dpsim[0:i]
    expref[i] = tmpexpref
    expref = expref[0:i]
    idx[i , :] = tmpidx
    idx = idx[0:i , :]
else:
    netsim = tmpnetsim
    dpsim = tmpnetsim - tmpexpref
    expref = tmpexpref
    idx = tmpidx
if PLT or DETAILS:
    print '\nNumber of identified clusters: %d\n' % K
    print 'Fitness (net similarity): %f\n' % tmpnetsim
    print '  Similarities of data points to exemplars: %f\n' % dpsim#[i-1]
    print '  Preferences of selected exemplars: %f\n' % tmpexpref
    print 'Number of iterations: %d\n\n' % i
if unconverged:
    print '\n*** Warning: Algorithm did not converge. The similarities\n'
    print '    may contain degeneracies - add noise to the similarities\n'
    print '    to remove degeneracies. To monitor the net similarity,\n'
    print '    activate plotting. Also, consider increasing maxits and\n'
    print '    if necessary dampfact.\n\n'

# Plot figure showing data and the clusters to compare with Matlab
#print 'Number of clusters: %d\n' % np.size(np.unique(idx))
if PLT:
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
