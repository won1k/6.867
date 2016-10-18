import numpy as np
from plotBoundary import *
import pylab as pl
from pegasos import kernelPegasos

def gausRBF(x1, x2, gamma = 1):
	# assume x are vectors of size p
	return np.exp(-gamma*np.linalg.norm(x1-x2)**2)

# load data from csv files
train = np.loadtxt('data/data3_train.csv')
X = train[:,0:2]
y = train[:,2:3]

# Carry out training.
max_epochs = len(y) * 20
lamb = .02
gammas = [2**i for i in range(-2,3)]
n = len(y)

# Support vectors
sv = []

for t, gamma in enumerate(gammas):
	# Compute K
	K = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			K[i][j] = gausRBF(X[i,:], X[j,:], gamma)

	# Train
	alpha = kernelPegasos(K, y, lamb, max_epochs)
	sv.append(sum([1 if a > 1e-5 else 0 for a in alpha]))

	# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
	def predict_gaussianSVM(x):
		pred = 0
		for i in range(n):
			pred += alpha[i] * y[i] * gausRBF(X[i,:], x, gamma)
		return(pred)

	# plot training results
	plotDecisionBoundary(X, y, predict_gaussianSVM, [-1,0,1], title = 'Gamma = ' + str(gamma))
	pl.savefig('hw2_3-4_' + str(t) + '.pdf')

np.savetxt('hw2_3-4_sv.txt', np.array(sv))
# Plot
pl.figure()
pl.plot(gammas, sv)
pl.savefig('hw2_3-4_sv.pdf')

