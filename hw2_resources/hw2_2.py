import svm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

def gausRBF(x1, x2, gamma = 1):
	# assume x are vectors of size p
	return np.exp(-gamma*np.linalg.norm(x1-x2)**2)

# Example
ex = np.loadtxt("data/svm_data.csv")
X = ex[:,0:-1]
y = ex[:,-1]
sol = svm.trainSVM(X, y, 1)

# Plot example
x1 = [x[0] for x in X]
x2 = [x[1] for x in X]
cols = ['b' if v > 0 else 'r' for v in y]
#marks = ['o' if v > 0 else 'x' for v in y]
plt.figure()
plt.scatter(x1, x2, color = cols)#, marker = marks)
plt.savefig('hw2-2_1-1.png')

# 2-3
name = '1'
train = np.loadtxt("data/data" + name + "_train.csv")
X_train = train[:,0:-1]
y_train = np.reshape(train[:,-1], (400,1))
n = len(y_train)
Cs = [0.01, 0.1, 1, 10, 100]
gammas = [10**i for i in range(-3, 3)]

lin_margins = []
lin_sv = []
gaus_margins = []
gaus_sv = []
for C in Cs:
	lin_sol = svm.trainSVM(X_train, y_train, C)
	lin_sv.append(sum([1 if a > 1e-5 else 0 for a in lin_sol['x']]))
	w = np.sum(lin_sol['x']*y_train*X_train,0)
	lin_margins.append(1/np.linalg.norm(w))

	for gamma in gammas:
		# Define kernel
		def kernel(x1, x2):
			return gausRBF(x1, x2, gamma)

		gaus_sol = svm.trainKernelSVM(kernel, X_train, y_train, C)
		gaus_sv.append(sum([1 if a > 1e-5 else 0 for a in gaus_sol['x']]))
		K = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				K[i][j] = gausRBF(X_train[i,:], X_train[j,:], gamma)
		ay = gaus_sol['x'] * y_train
		w_norm = np.dot(np.transpose(ay), np.dot(K, ay))
		gaus_margins.append(1/np.sqrt(w_norm))

# Save
np.savetxt("hw2-2_3_linmargins.txt", np.array(lin_margins))
np.savetxt("hw2-2_3_linsv.txt", np.array(lin_sv))
np.savetxt("hw2-2_3_gausmargins.txt", np.array(gaus_margins))
np.savetxt("hw2-2_3_gaussv.txt", np.array(gaus_sv))

# Plot
plt.figure()
plt.plot(np.log(Cs), lin_margins)
plt.xlabel("Log(C)")
plt.ylabel("Margin Size")
plt.savefig('hw2-2_3_linmargins.pdf')

plt.figure()
plt.plot(np.log(Cs), lin_sv)
plt.xlabel("Log(C)")
plt.ylabel("Support Vectors")
plt.savefig('hw2-2_3_linsv.pdf')

plotC = [[np.log(C)]*len(gammas) for C in Cs]
plotC = [item for sublist in plotC for item in sublist]
plotGammas = np.log(gammas*len(Cs))
gaus_margins = [margin[0][0] for margin in gaus_margins]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(plotGammas, plotC, gaus_margins)
ax.set_xlabel('Log(Gamma)')
ax.set_ylabel('Log(C)')
ax.set_zlabel('Margin Size')
plt.savefig('hw2-2_3_gausmargins.pdf')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(plotGammas, plotC, gaus_sv)
ax.set_xlabel('Log(Gamma)')
ax.set_ylabel('Log(C)')
ax.set_zlabel('Support Vectors')
plt.savefig('hw2-2_3_gaussv.pdf')
