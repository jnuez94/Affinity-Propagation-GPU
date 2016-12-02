import numpy as np
import utility.py as ut

"""
Python version of apclusterSparse.m
utility.py reads data and generates float32 numpy array of coordinates
Sparse similarity matrix is generated here
"""
# Read point cloud
data = utility.readPointCloud('.xyz')
if np.size(data, 0) != 3: print 'Data should be a 3D point cloud'
if data.dtype != np.float32: print 'Data needs to be float32'
# Generate sparse similarity matrix
N = data.shape[1]
M = N*N-N
s = np.zeros((3,M))
j = 1
for i in range(N):
    for k in range(N):
        if k == i: continue
        s[0][j] = i
        s[1][j] = k
        s[2][j] = -np.sum(np.square(data[:][i] - data[:][k]))
        j += 1
# Set preference to median similarity
p = np.median(s[2][:])

#PARAMETERS
#defaults for now
MAXITS = 1000 # maximum iterations. Default = 1000
CONVITS = 100 # converged if est. centers stay fixed for convits iterations. Default = 100
DAMPFACT = 0.9 # 0.5 to 1, damping. Higher needed if oscillations occur. Default = 0.9.
PLT = False # Plots net similarity after each iteration
DETAILS = False # Store idx, netsim, dpsim, expref after each iteration
NONOISE = False # Degenerate input similarities. True = noise removal.
if lam>0.9:
    print 'Large damping factor, turn on plotting! Consider using larger value of convits.'

########################
# BEGIN MEAT OF FUNCTION
########################

# Make vector of preferences
if np.size(p, 0) == 1: p = p*np.ones(N, np.float32)

# Append self-similarities (preferences) to s-matrix
tmps = np.vstack((np.tile(range(1,N+1), (2,1)), p))
s = np.vstack((s, tmps))

# Add a small amount of noise to input similarities
# TODO: figure out how do do the randomness

# Construct indices of neighbors
ind1e = np.zeros(N)
for j in range(M):
    k = s[0][j]
    ind1e[k] += 1
ind1e = np.cumsum(ind1e)
ind1s = np.concatenate(np.ones(1), ind1e[0:-1] + 1)
ind1 = np.zeros(M)
for j in range(M):
    k = s[0][j]
    ind1[ind1s[k]] = j
    ind1s[k] += 1
ind1s = np.concatenate(np.ones(1), ind1e[0:-1] + 1)
ind2e = np.zeros(N)
for j in range(M):
    k = s[1][j]
    ind2e[k] += 1
ind2e = np.cumsum(ind2e)
ind2s = np.concatenate(np.ones(1), ind2e[0:-1] + 1)
ind2 = np.zeros(M)
for j in range(M):
    k = s[1][j]
    ind2[ind2[k]] = j
    ind2s[k] += 1
ind2s = np.concatenate(np.ones(1), ind2s[0:-1] + 1)

# Allocate space for messages
A = np.zeros(M)
R = np.zeros(M)
t = 1
if PLT: netsim = np.zeros((MAXITS+1, 1))
if DETAILS:
    idx = np.zeros((MAXITS+1, N))
    netsim = np.zeros((MAXITS+1, 1))
    dpsim = np.zeros((MAXITS+1, 1))
    expref = np.zeros((MAXITS+1, 1))

# Execute parallel affinity propagation updates
e = np.zeros((CONVITS, N))
dn = 0
i = 0
while not dn:
    i += 1

    # Compute responsibilities
    for j in range(N):
        ss = s[2][ind1[ind1s[j]] : ind1e[j]]
        as = A[ind1[ind1s[j]] : ind1e[j]] + ss
        # TODO: replace as variable, keep going!
