import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Data
data = np.genfromtxt("hw3_2-5_fs.txt", delimiter = ",")
fsize = data[:,0]
stride = data[:,1]
acc = data[:,7]
max_acc = data[:,8]

print data[np.argmax(max_acc),:]

# Plot
fig = pl.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(fsize, stride, max_acc)
#ax.plot_trisurf(fsize, stride, max_acc, color = 'red')
ax.set_xlabel('Filter Size')
ax.set_ylabel('Stride')
ax.set_zlabel('Valid. Acc.')
pl.show()
#pl.savefig('hw3_2-5_small.pdf')