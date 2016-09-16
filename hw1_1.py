'''
Homework 1, 6.867
Problem 1
Won Lee
'''

import numpy as np
import pylab as pl
import scipy.stats as stat
import matplotlib.pyplot as plt

def getData():
    # load the parameters for the negative Gaussian function and quadratic bowl function
    # return a tuple that contains parameters for Gaussian mean, Gaussian covariance,
    # A and b for quadratic bowl in order
    data = pl.loadtxt('code/P1/parametersp1.txt')
    gaussMean = data[0,:]
    gaussCov = data[1:3,:]
    quadBowlA = data[3:5,:]
    quadBowlb = data[5,:]
    return (gaussMean, gaussCov, quadBowlA, quadBowlb) 

# 1.1

# Generic gradient descent
def gradientDescent(obj, grad, init, step, eps):
	prev_loss = 1e9
	curr_loss = 1e8
	x = init
	while abs(curr_loss - prev_loss) >= eps:
		x = x - step * grad(x)
		prev_loss = curr_loss
		curr_loss = obj(x)
		#print "Current loss: " + str(curr_loss)
	return x

gaussMean, gaussCov, quadBowlA, quadBowlb = getData()

# Objective and gradients
def negGauss(x):
	n = len(x)
	return -np.exp(-0.5*np.dot(x-gaussMean,np.dot(np.linalg.inv(gaussCov),x-gaussMean)))/((2*np.pi)**(n/2)*np.linalg.det(gaussCov))

def negGaussGrad(x):
	return -negGauss(x)*np.dot(np.linalg.inv(gaussCov), (x - gaussMean))

def quadBowl(x):
	return 0.5*np.dot(x, np.dot(quadBowlA, x)) - np.dot(x, quadBowlb)	

def quadBowlGrad(x):
	return np.dot(quadBowlA, x) - quadBowlb

# NegGaussian: Run for init, step, eps
inits = [np.array(x) for x in [[1,1],[5,5],[1,9],[20,15]]]
steps = [10**i for i in range(5,10)]
epss = [10**(-i) for i in range(10,20)]
x = []

for init in inits:
	for step in steps:
		for eps in epss:
			print(init, step, eps)
			x.append(gradientDescent(negGauss, negGaussGrad, init, step, eps))

sols = []
for i, init in enumerate(inits):
	sols.append(x[50*i:50*(i+1)])

dist = []
for points in sols:
	dist_point = []
	for point in points:
		dist_point.append(np.linalg.norm(point - gaussMean))
	dist.append(dist_point)

from mpl_toolkits.mplot3d import Axes3D
for i, x in enumerate(inits):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_trisurf(np.repeat([np.log(step) for step in steps], 10), [np.log(eps) for eps in epss]*5, dist[i])
	ax.set_xlabel('Log(Step Size)')
	ax.set_ylabel('Log(Epsilon)')
	ax.set_zlabel('Squared Error')
	plt.savefig('hw1_1-1_' + str(i) + '.pdf')
plt.show()

# QuadBowl: Run for init, step, eps
exact = np.dot(np.linalg.inv(quadBowlA), quadBowlb)
inits = [np.array(x) for x in [[1,1],[5,5],[1,9],[20,15]]]
steps = [10**(-i) for i in range(1,6)]
epss = [10**(-i) for i in range(10,20)]
x = []

for init in inits:
	for step in steps:
		for eps in epss:
			print(init, step, eps)
			x.append(gradientDescent(quadBowl, quadBowlGrad, init, step, eps))

sols = []
for i, init in enumerate(inits):
	sols.append(x[50*i:50*(i+1)])

dist = []
for points in sols:
	dist_point = []
	for point in points:
		dist_point.append(np.linalg.norm(point - exact))
	dist.append(dist_point)

for i, x in enumerate(inits):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_trisurf(np.repeat([np.log(step) for step in steps], 10), [np.log(eps) for eps in epss]*5, dist[i])
	ax.set_xlabel('Log(Step Size)')
	ax.set_ylabel('Log(Epsilon)')
	ax.set_zlabel('Squared Error')
	plt.savefig('hw1_1-2_' + str(i) + '.pdf')
plt.show()

# 1.2

def finiteDiff(obj, diff_size, x):
	d = len(x)
	finiteGrad = np.zeros(d)
	for i in range(d):
		unit = np.zeros(d)
		unit[i] = diff_size
		finiteGrad[i] = (obj(x + unit) - obj(x - unit))/(2*diff_size)
	return finiteGrad

diff_sizes = [10**(-i) for i in range(5)]
dist = []
for init in inits:
	dist_point = []
	for diff_size in diff_sizes:
		dist_point.append(np.linalg.norm(finiteDiff(negGauss, diff_size, init) - negGaussGrad(init)))
	dist.append(dist_point)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(4):
	ax.plot([np.log(diff) for diff in diff_sizes], [np.log(d) for d in dist[i]])


ax.set_xlabel('Log(Delta)')
ax.set_ylabel('Log(Squared Error)')
plt.savefig('hw1_1-2a.pdf')
plt.show()

dist = []
for init in inits:
	dist_point = []
	for diff_size in diff_sizes:
		dist_point.append(np.linalg.norm(finiteDiff(quadBowl, diff_size, init) - quadBowlGrad(init)))
	dist.append(dist_point)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(4):
	ax.plot([np.log(diff) for diff in diff_sizes], dist[i])


ax.set_xlabel('Log(Delta)')
ax.set_ylabel('Squared Error')
plt.savefig('hw1_1-2b.pdf')
plt.show()

# 1.3

def getData():
    
    # load the fitting data for X and y and return as elements of a tuple
    # X is a 100 by 10 matrix and y is a vector of length 100
    # Each corresponding row for X and y represents a single data sample

    X = pl.loadtxt('code/P1/fittingdatap1_x.txt')
    y = pl.loadtxt('code/P1/fittingdatap1_y.txt')

    return (X,y) 

X, y = getData()

def costFunction(theta):
	return np.linalg.norm(np.dot(X, theta) - y)

def costFuncGrad(theta):
	return 2*np.dot(np.transpose(X), np.dot(X, theta) - y)

def costFuncGradPoint(theta, x, y):
	return 2*(np.dot(x, theta) - y)*x

def SGD(obj, grad, init, step, eps, tau, kappa):
	n = len(y)
	prev_loss = 1e9
	curr_loss = 1e8
	theta = init
	t = 1
	while abs(curr_loss - prev_loss) >= eps:
		step = step/(tau + t)**kappa
		idx = np.random.randint(0, n, 1)[0]
		theta = theta - step * grad(theta, X[idx,:], y[idx])
		t += 1
		prev_loss = curr_loss
		curr_loss = obj(theta)
		#print(curr_loss)
	return theta, t

def gradientDescent(obj, grad, init, step, eps, tau, kappa):
	prev_loss = 1e9
	curr_loss = 1e8
	x = init
	t = 1
	while abs(curr_loss - prev_loss) >= eps:
		step = step/(tau + t)**kappa
		x = x - step * grad(x)
		t += 1
		prev_loss = curr_loss
		curr_loss = obj(x)
		#print "Current loss: " + str(curr_loss)
	return x, t


def solveLS():
	XtX = np.dot(np.transpose(X), X)
	return np.dot(np.linalg.inv(XtX), np.dot(np.transpose(X), y))

def L2(x, y):
	return np.linalg.norm(x-y)

# Exact solution
exact = solveLS()
n, d = X.shape

init = np.ones(d)
step = 1e-5
eps = 1e-15
taus = [i*5 for i in range(10)]
kappas = [10**(-i) for i in range(6)]
accGD = []
evalsGD = []
accSGD = []
evalsSGD = []

for tau in taus:
	for kappa in kappas:
		print(tau, kappa)
		gradSol, gradT = gradientDescent(costFunction, costFuncGrad, init, step, eps, tau, kappa)
		accGD.append(L2(gradSol, exact))
		evalsGD.append(gradT * n)
		sgSol, sgT = SGD(costFunction, costFuncGradPoint, init, step, eps, tau, kappa)
		accSGD.append(L2(sgSol, exact))
		evalsSGD.append(sgT)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(np.repeat(taus, 6), [np.log(kappa) for kappa in kappas]*10, accGD)
ax.plot_trisurf(np.repeat(taus, 6), [np.log(kappa) for kappa in kappas]*10, accSGD, color = 'r')
ax.set_xlabel('Tau')
ax.set_ylabel('Log(Kappa)')
ax.set_zlabel('Squared Error')
plt.savefig('hw1_1-3_acc.pdf')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(np.repeat(taus, 6), [np.log(kappa) for kappa in kappas]*10, evalsGD)
#ax.plot_trisurf(np.repeat(taus, 6), [np.log(kappa) for kappa in kappas]*10, evalsSGD, color = 'r')
ax.set_xlabel('Tau')
ax.set_ylabel('Log(Kappa)')
ax.set_zlabel('Number of Evaluations')
plt.savefig('hw1_1-3_timeGD.pdf')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(np.repeat(taus, 6), [np.log(kappa) for kappa in kappas]*10, evalsGD)
ax.plot_trisurf(np.repeat(taus, 6), [np.log(kappa) for kappa in kappas]*10, evalsSGD, color = 'r')
ax.set_xlabel('Tau')
ax.set_ylabel('Log(Kappa)')
ax.set_zlabel('Number of Evaluations')
plt.savefig('hw1_1-3_timeSGD.pdf')
plt.show()








