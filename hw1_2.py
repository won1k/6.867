'''
Homework 1, 6.867
Problem 2
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

# 2.1 

def cosFunc(x):
	return np.cos(np.pi*x) + 1.5*np.cos(2*np.pi*x)

def polyFunc(x, M, weights):
	phi = np.array([x**m for m in range(M + 1)])
	return np.dot(weights, phi)

def polyMLE(X, Y, M):
	phi = []
	for i in range(M + 1):
		phi.append([x**i for x in X])
	phi = np.transpose(np.array(phi))
	phiTphi = np.dot(np.transpose(phi), phi)
	return np.dot(np.linalg.inv(phiTphi), np.dot(np.transpose(phi), Y))

Ms = [0, 1, 3, 10]
x_points = [float(x)/10000 for x in range(10000)]
y_points = [cosFunc(x) for x in x_points]
for M in Ms:
	fig = plt.figure()
	weightMLE = polyMLE(X, Y, M)
	plt.plot(X, Y, 'o')
	est_points = [polyFunc(x, M, weightMLE) for x in x_points]
	plt.plot(x_points, y_points)
	plt.plot(x_points, est_points)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('hw1_2-1_' + str(M) + '.pdf')
plt.show()


# 2.2

def SSE(X, Y, weights):
	M = len(weights) - 1
	phi = []
	for i in range(M + 1):
		phi.append([x**i for x in X])
	phi = np.transpose(np.array(phi))
	return np.linalg.norm(np.dot(phi, weights) - Y)**2

def gradSSE(X, Y, weights):
	M = len(weights) - 1
	phi = []
	for i in range(M + 1):
		phi.append([x**i for x in X])
	phi = np.transpose(np.array(phi))
	return 2*np.dot(np.transpose(phi), np.dot(phi, weights) - Y)

def finiteDiff(obj, diff_size, w):
	d = len(w)
	finiteGrad = np.zeros(d)
	for i in range(d):
		unit = np.zeros(d)
		unit[i] = diff_size
		finiteGrad[i] = (obj(X, Y, w + unit) - obj(X, Y, w - unit))/(2*diff_size)
	return finiteGrad

weights = [[1,1], [5,3,4], [-5,5,-5,5], [1,1,1,1,1,1,1,1,-1]]
diff_size = 0.1
for weight in weights:
	np.linalg.norm(gradSSE(X, Y, weight) - finiteDiff(SSE, diff_size, weight))




