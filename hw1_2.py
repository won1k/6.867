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

# 2.3

def gradientDescent(obj, grad, init, step, eps):
	prev_loss = 1e9
	curr_loss = 1e8
	w = init
	t = 1
	while abs(curr_loss - prev_loss) >= eps:
		w = w - step * grad(X, Y, w)
		t += 1
		prev_loss = curr_loss
		curr_loss = obj(X, Y, w)
		#print "Current loss: " + str(curr_loss)
	return w, t

def SGD(obj, grad, init, step, eps, tau, kappa, max_iter):
	n = len(X)
	prev_loss = 1e9
	curr_loss = 1e8
	w = init
	t = 1
	while abs(curr_loss - prev_loss) >= eps and t < max_iter:
		step = step/(tau + t)**kappa
		idx = np.random.randint(0, n, 1)[0]
		w = w - step * grad(X, Y, w)
		t += 1
		prev_loss = curr_loss
		curr_loss = obj(X, Y, w)
		print(curr_loss)
	return w, t

Ms = [1, 3, 5]
inits = []
for M in Ms:
	inits += [np.array([1]*(M+1)), np.array([5]*((M+1)/2) + [-10]*((M+1)/2))]
steps = [10**(-i-1) for i in range(3)]
epss = [10**(-2*i) for i in range(2,5)]
kappa = 1e-5
tau = 1
max_iter = 1e8

exacts = {}
GDsols = {}
GDevals = {}
SGDsols = {}
SGDevals = {}
for init in inits:
	i = tuple(list(init))
	exacts[i] = []
	GDsols[i] = []
	GDevals[i] = []
	SGDsols[i] = []
	SGDevals[i] = []
	for step in steps:
		for eps in epss:
			print(init, step, eps)
			exacts[i].append(polyMLE(X, Y, len(init) - 1))
			GDsol, GDtime = gradientDescent(SSE, gradSSE, init, step, eps)
			GDsols[i].append(GDsol)
			GDevals[i].append(GDtime)
			SGDsol, SGDtime = SGD(SSE, gradSSE, init, step, eps, tau, kappa, max_iter)
			SGDsols[i].append(SGDsol)
			SGDevals[i].append(SGDtime)

GDdists = {}
SGDdists = {}
for key, val in exacts.iteritems():
	GDdists[key] = []
	SGDdists[key] = []
	for i, exact in enumerate(val):
		GDdists[key].append(np.linalg.norm(GDsols[key][i]- exact))
		SGDdists[key].append(np.linalg.norm(SGDsols[key][i] - exact))

from mpl_toolkits.mplot3d import Axes3D
for i, x in enumerate(inits):
	x = tuple(list(x))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	GDheights = [h if h < 260 else 260 for h in GDdists[x]]
	SGDheights = [h if h < 260 else 260 for h in SGDdists[x]]
	ax.plot_trisurf(np.repeat([np.log(step) for step in steps], 3), [np.log(eps) for eps in epss]*3, GDheights)
	ax.plot_trisurf(np.repeat([np.log(step) for step in steps], 3), [np.log(eps) for eps in epss]*3, SGDheights, color = 'r')
	ax.set_xlabel('Log(Step Size)')
	ax.set_ylabel('Log(Epsilon)')
	ax.set_zlabel('Squared Error')
	plt.savefig('hw1_2-3_' + str(i) + '.pdf')
plt.show()

# 2.4

def cosMLE(X, Y, M):
	phi = []
	for i in range(1,M+1):
		phi.append([np.cos(i*np.pi*x) for x in X])
	phi = np.transpose(np.array(phi))
	phiTphi = np.dot(np.transpose(phi), phi)
	return np.dot(np.linalg.inv(phiTphi), np.dot(np.transpose(phi), Y))

def cosRegFunc(x, weights):
	M = len(weights)
	phi = np.array([np.cos(m*np.pi*x) for m in range(1, M+1)])
	return np.dot(phi, weights)

cosWeightMLE = cosMLE(X, Y, 8) # [0.769, 1.087, 0.0993, 0.143, -0.0507, 0.362, 0.0123, 0.0151]
x_points = [float(x)/10000 for x in range(10000)]
y_points = [cosFunc(x) for x in x_points]
est_points = [cosRegFunc(x, cosWeightMLE) for x in x_points]
fig = plt.figure()
plt.plot(X, Y, 'o')
plt.plot(x_points, y_points)
plt.plot(x_points, est_points)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('hw1_2-4.pdf')
plt.show()






