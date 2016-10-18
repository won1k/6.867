import numpy as np
from plotBoundary import *
import pylab as pl
from pegasos import pegasos

# load data from csv files
train = np.loadtxt('data/data3_train.csv')
X = train[:,0:2]
y = train[:,2:3]

# Carry out training.
lambdas = [2**(-i) for i in range(-1,11)]
max_epochs = len(y) * 20
w_all = np.zeros((len(lambdas), X.shape[1]))
w0_all = np.zeros(len(lambdas))

for i, lamb in enumerate(lambdas):
	w_all[i,:], w0_all[i] = pegasos(X, y, lamb, max_epochs)

# Save output
np.savetxt("hw2_3-2_w.txt", w_all)
np.savetxt("hw2_3-2_w0.txt", w0_all)

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
for i, lamb in enumerate(lambdas):
	w = w_all[i,:]
	w0 = w0_all[i]
	def predict_linearSVM(x):
		pred = np.dot(w,x) + w0
		return(1 if pred > 0 else -1)

	# plot training results
	plotDecisionBoundary(X, y, predict_linearSVM, [-1,0,1], title = 'Lambda = ' + str(lamb))
	pl.savefig('hw2_3-2_'+str(i)+'.pdf')

margins = [1/np.linalg.norm(w) for w in w_all[1:,]]
pl.figure()
pl.plot(lambdas[1:], margins)
axes = pl.gca()
axes.set_ylim([0,1.2])
pl.savefig('hw2_3-2_margins.pdf')