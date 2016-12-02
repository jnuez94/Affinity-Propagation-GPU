#!/usr/bin/env python

import numpy as np

# Parameter: filename of the data set
# Return: x, y, z numpy arrays.
def readPointCloud(filename):
	x = []
	y = []
	z = []
	for lines in open(filename).readlines():
		xi, yi, zi, _, _ = lines.split(' ')
		x.append(float(xi))
		y.append(float(yi))
		z.append(float(zi))

	x = np.copy(x).astype(np.float32)
	y = np.copy(y).astype(np.float32)
	z = np.copy(z).astype(np.float32)
	return x, y, z

