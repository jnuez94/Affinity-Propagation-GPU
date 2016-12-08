#!/usr/bin/env python

import numpy as np

# Parameter: filename of the data set
# Return: x, y, z numpy arrays.
def readPointCloud(filename, size):
	x = []
	y = []
	z = []
	data = open(filename)
	for i, lines in enumerate(data):
		if i == size:
			break
		xi, yi, zi, _, _ = lines.split(' ')
		x.append(float(xi))
		y.append(float(yi))
		z.append(float(zi))

	x = np.copy(x).astype(np.float32)
	y = np.copy(y).astype(np.float32)
	z = np.copy(z).astype(np.float32)
	return x, y, z

def pysimilarity(x, y, z):
	size = len(x)
	sim = np.zeros((size,size), np.float32)
	for i in xrange(0, size, 1):
		for j in xrange(0, size, 1):
			sim[i][j] = -((x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2)

	return sim