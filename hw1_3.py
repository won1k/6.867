'''
Homework 1, 6.867
Problem 3
Won Lee
'''

import numpy as np
import pylab as pl
import scipy.stats as stat
import matplotlib.pyplot as plt

def getData(ifPlotData = False):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple
    data = pl.loadtxt('code/P2/curvefittingp2.txt')
    X = data[0,:]
    Y = data[1,:]
    if ifPlotData:
        plt.plot(X,Y,'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    return (X,Y)

X,Y = getData()

def L2(x, y):
	return np.linalg.norm(x-y)

# 3.1

def cosFunc(x):
	return np.cos(np.pi*x) + 1.5*np.cos(2*np.pi*x)

def polyRidge(X, Y, M, lamb):
	phi = []
	for i in range(M + 1):
		phi.append([x**i for x in X])
	phi = np.transpose(np.array(phi))
	invMat = np.linalg.inv(np.dot(np.transpose(phi), phi) + lamb*np.eye(M + 1))
	return np.dot(invMat, np.dot(np.transpose(phi), [y for y in Y]))

def polyFunc(x, M, weights):
	phi = np.array([x**m for m in range(M + 1)])
	return np.dot(weights, phi)

Ms = [3,5,8,10]
lambs = [0, 0.1, 0.5, 1, 10]
weights = []
x_points = [float(x)/10000 for x in range(10000)]
y_points = [cosFunc(x) for x in x_points]
for M in Ms:
	fig = plt.figure()
	plt.plot(X, Y, 'o')
	plt.plot(x_points, y_points, label = "true function")
	plt.xlabel('x')
	plt.ylabel('y')
	for lamb in lambs:
		weight = polyRidge(X, Y, M, lamb)
		weights.append(weight)
		est_points = [polyFunc(x, M, weight) for x in x_points]
		plt.plot(x_points, est_points, label = str(lamb))
	plt.legend()
	plt.savefig('hw1_3-1_' + str(M) + '.pdf')
	#plt.show()

# 3.2

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def regressAData():
    return getData('code/P3/regressA_train.txt')

def regressBData():
    return getData('code/P3/regressB_train.txt')

def validateData():
    return getData('code/P3/regress_validate.txt')

X_A, Y_A = regressAData()
X_B, Y_B = regressBData()
X_V, Y_V = validateData()

# Use A as train
Ms = [1, 3, 5, 7]
lambs = [0, 0.1, 0.5, 1, 5, 10]
weights = {}
valid_loss = {}
for M in Ms:
	for lamb in lambs:
		weight = polyRidge(X_A, Y_A, M, lamb)
		weights[(M, lamb)] = weight
		pred = np.array([[polyFunc(x[0], M, weight)] for x in X_V])
		valid_loss[(M,lamb)] = L2(Y_V, pred)
M_A, lamb_A = min(valid_loss, key = valid_loss.get)
# Plot for B (test)
x_points = [float(x)/1000 for x in range(-3000,2000)]
fig = plt.figure()
plt.plot(X_B, Y_B, 'o')
plt.xlabel('x')
plt.ylabel('y')
for M in Ms:
	for lamb in lambs:
		weight = weights[(M, lamb)]
		est_points = [polyFunc(x, M, weight) for x in x_points]
		if M == M_A and lamb == lamb_A:
			plt.plot(x_points, est_points, label = str(lamb), lw = 2)
		else:
			plt.plot(x_points, est_points, label = str(lamb), ls = "dotted", lw = 0.5)
plt.savefig('hw1_3-2_A.pdf')
# Plot squared errors
from mpl_toolkits.mplot3d import Axes3D
plotM = [M for (M,lamb) in valid_loss.keys()]
plotLamb = [lamb for (M,lamb) in valid_loss.keys()]
plotLoss = valid_loss.values()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(plotM, plotLamb, plotLoss)
ax.set_xlabel('M')
ax.set_ylabel('Lambda')
ax.set_zlabel('Squared Error')
plt.savefig('hw1_3-2_A_err.pdf')
plt.show()

# Use B as train
Ms = [1, 3, 5, 7]
lambs = [0, 0.1, 0.5, 1, 5, 10]
weights = {}
valid_loss = {}
for M in Ms:
	for lamb in lambs:
		weight = polyRidge(X_B, Y_B, M, lamb)
		weights[(M, lamb)] = weight
		pred = np.array([[polyFunc(x[0], M, weight)] for x in X_V])
		valid_loss[(M,lamb)] = L2(Y_V, pred)
M_B, lamb_B = min(valid_loss, key = valid_loss.get)
# Plot for A (test)
x_points = [float(x)/1000 for x in range(-3000,2500)]
fig = plt.figure()
plt.plot(X_A, Y_A, 'o')
plt.xlabel('x')
plt.ylabel('y')
for M in Ms:
	for lamb in lambs:
		weight = weights[(M, lamb)]
		est_points = [polyFunc(x, M, weight) for x in x_points]
		if M == M_B and lamb == lamb_B:
			plt.plot(x_points, est_points, label = str(lamb), lw = 2)
		else:
			plt.plot(x_points, est_points, label = str(lamb), ls = "dotted", lw = 0.5)
plt.savefig('hw1_3-2_B.pdf')
# Plot squared errors
from mpl_toolkits.mplot3d import Axes3D
plotM = [M for (M,lamb) in valid_loss.keys()]
plotLamb = [lamb for (M,lamb) in valid_loss.keys()]
plotLoss = valid_loss.values()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(plotM, plotLamb, plotLoss)
ax.set_xlabel('M')
ax.set_ylabel('Lambda')
ax.set_zlabel('Squared Error')
plt.savefig('hw1_3-2_B.err.pdf')
plt.show()



