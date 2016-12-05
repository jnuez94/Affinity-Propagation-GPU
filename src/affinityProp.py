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
# Generate sparse similarity matrix
N = data.shape[1]
M = N*N-N
s = np.zeros((3,M), np.float32)
j = 0
for i in range(N):
    for k in range(N):
        if k == i: continue
        s[0][j] = i
        s[1][j] = k
        s[2][j] = -np.sum(np.square(data[:,i] - data[:,k]))
        j += 1
# Set preference to median similarity
p = np.median(s[2,:]).astype(np.float32)

#PARAMETERS
#defaults for now
MAXITS = 1000 # maximum iterations. Default = 1000
CONVITS = 100 # converged if est. centers stay fixed for convits iterations. Default = 100
DAMPFACT = 0.9 # 0.5 to 1, damping. Higher needed if oscillations occur. Default = 0.9.
PLT = True # Plots net similarity after each iteration
DETAILS = False # Store idx, netsim, dpsim, expref after each iteration
NONOISE = True # Degenerate input similarities. True = noise removal.
if DAMPFACT>0.9:
    print 'Large damping factor, turn on plotting! Consider using larger value of convits.'

########################
# BEGIN MEAT OF FUNCTION
########################

# Make vector of preferences
if np.size(p) == 1: p = p*np.ones(N, np.float32)

# Append self-similarities (preferences) to s-matrix
tmps = np.vstack((np.tile(range(N), (2,1)), p)).astype(np.float32)
s = np.hstack((s, tmps))
M = np.shape(s)[1]

# Add a small amount of noise to input similarities
if not NONOISE:
    np.random.seed()
    s[2,0:M] += np.multiply(np.finfo('float32').eps * s[2,0:M] + np.finfo('float32').tiny * 100 , np.random.rand(M))

# Construct indices of neighbors
ind1e = np.zeros(N, np.int32)
for j in range(M):
    k = int(s[0,j])
    ind1e[k] += 1
ind1e = np.cumsum(ind1e).astype(np.int32)
ind1s = np.concatenate((np.zeros(1, np.int32), ind1e[0:-1]))
ind1 = np.zeros(M, np.int32)
for j in range(M):
    k = int(s[0,j])
    ind1[ind1s[k]] = j
    ind1s[k] += 1
ind1s = np.concatenate((np.zeros(1, np.int32), ind1e[0:-1]))
ind2e = np.zeros(N, np.int32)
for j in range(M):
    k = int(s[1,j])
    ind2e[k] += 1
ind2e = np.cumsum(ind2e).astype(np.int32)
ind2s = np.concatenate((np.zeros(1, np.int32), ind2e[0:-1]))
ind2 = np.zeros(M, np.int32)
for j in range(M):
    k = int(s[1,j])
    ind2[ind2s[k]] = j
    ind2s[k] += 1
ind2s = np.concatenate((np.zeros(1, np.int32), ind2e[0:-1]))

# Allocate space for messages
A = np.zeros(M, np.float32)
R = np.zeros(M, np.float32)
t = 1
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
    for j in range(N):
        ss = s[2 , ind1[ind1s[j] : ind1e[j]]]
        a_s = A[ind1[ind1s[j] : ind1e[j]]] + ss
        Y = np.max(a_s).astype(np.float32)
        I = np.argmax(a_s)
        a_s[I] = np.finfo('float32').min
        Y2 = np.max(a_s).astype(np.float32)
        I2 = np.argmax(a_s)
        r = ss - Y
        r[I] = ss[I] - Y2
        R[ind1[ind1s[j] : ind1e[j]]] = (1-DAMPFACT) * r + DAMPFACT * R[ind1[ind1s[j] : ind1e[j]]]

    # Compute availabilities
    for j in range(N):
        rp = R[ind2[ind2s[j] : ind2e[j]]]
        rp[0:-1] = np.maximum(rp[0:-1], 0) # elementwise maximum!
        a = np.sum(rp) - rp
        a[0:-1] = np.minimum(a[0:-1], 0) # elementwise minimum!
        A[ind2[ind2s[j] : ind2e[j]]] = (1-DAMPFACT) * a + DAMPFACT * A[ind2[ind2s[j] : ind2e[j]]]

    # Check for convergence
    E = (A[M-N::] + R[M-N::]) > 0
    #print np.sum(E)
    e[(i-1) % CONVITS , :] = E
    K = np.sum(E).astype(np.int32)
    if i >= CONVITS or i>= MAXITS:
        se = np.sum(e, 0).astype(np.int32)
        unconverged = np.sum((se==CONVITS) + (se==0)) != N
        if (not unconverged and K>0) or (i==MAXITS):
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
                ss = s[2 , ind1[ind1s[j] : ind1e[j]]]
                ii = s[1 , ind1[ind1s[j] : ind1e[j]]].astype(np.int32)
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
plt.ion()
plt.show()

# Identify exemplars
E = ((A[M-N:M] + R[M-N:M]) > 0)
K = np.sum(E).astype(np.int32)
if K>0:
    tmpidx=np.zeros(N, np.float32)
    tmpidx[np.argwhere(E)] = np.argwhere(E)
    for j in np.argwhere(E==0).flatten():
        ss = s[2 , ind1[ind1s[j] : ind1e[j]]]
        ii = s[1 , ind1[ind1s[j] : ind1e[j]]].astype(np.int32)
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
            mm = s[1 , ind1[ind1s[m] : ind1e[m]]].astype(np.int32)
            msk[mm] += 1
            ns[mm] += s[2 , ind1[ind1s[m] : ind1e[m]]]
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
        ss = s[2 , ind1[ind1s[j] : ind1e[j]]]
        ii = s[1 , ind1[ind1s[j] : ind1e[j]]].astype(np.int32)
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
# Plot figure showing data and the clusters
#print 'Number of clusters: %d\n' % np.size(np.unique(idx))
if PLT:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure().gca(projection='3d')
    for i in np.unique(idx):
        ii = np.argwhere(idx == i)
        h = fig.scatter(data[0,ii], data[1,ii], zs=data[2,ii])
        plt.hold(True)
        col = np.tile(np.random.rand(1,3), (np.size(ii), 1))
        print col
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
