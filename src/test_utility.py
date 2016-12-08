import utility as ut

x, y, z = ut.readPointCloud('./data/data.xyz', 2048)

print "x: ", x
print "y: ", y
print "z: ", z
print "size: ", len(x)
