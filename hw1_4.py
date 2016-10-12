'''
Homework 1, 6.867
Problem 4
Won Lee
'''

import numpy as np
import pylab as pl
import scipy.stats as stat
import matplotlib.pyplot as plt

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def lassoTrainData():
    return getData('code/P4/lasso_train.txt')

def lassoValData():
    return getData('code/P4/lasso_validate.txt')

def lassoTestData():
    return getData('code/P4/lasso_test.txt')

# Load data and preprocess
X_train, Y_train = lassoTrainData()
X_train = [x[0] for x in X_train]
Y_train = [y[0] for y in Y_train]
phi_train = []
for i in range(M + 1):
	if i == 0:
		phi_train.append([x for x in X_train])
	else:
		phi_train.append([np.sin(0.4*np.pi*x*i) for x in X_train])
phi_train = np.transpose(np.array(phi_train))
X_val, Y_val = lassoValData()
X_val = [x[0] for x in X_val]
Y_val = [y[0] for y in Y_val]
phi_val = []
for i in range(M + 1):
	if i == 0:
		phi_val.append([x for x in X_val])
	else:
		phi_val.append([np.sin(0.4*np.pi*x*i) for x in X_val])
phi_val = np.transpose(np.array(phi_val))
X_test, Y_test = lassoTestData()
X_test = [x[0] for x in X_test]
Y_test = [y[0] for y in Y_test]
phi_test = []
for i in range(M + 1):
	if i == 0:
		phi_test.append([x for x in X_test])
	else:
		phi_test.append([np.sin(0.4*np.pi*x*i) for x in X_test])
phi_test = np.transpose(np.array(phi_test))

# Model selection on various lambda
lambs = [1e-3, 0.1, 0.5, 1, 5, 10]
M = 12
true_weight = pl.loadtxt('code/P4/lasso_true_w.txt')
eps = 1e-15
max_iter = 1e4
init = np.ones(M+1)
step = 1e-3

def L2(x, y):
	return np.linalg.norm(x-y)

# Fit MLE
def MLE(Phi, Y):
	matInv = np.linalg.inv(np.dot(Phi, np.transpose(Phi)))
	return np.dot(np.transpose(Phi), np.dot(matInv, Y))

MLE_weight = MLE(phi_train, Y_train)

# Fit ridge
def SSE(Phi, Y, weights):
	return 0.5*L2(np.dot(Phi, weights),Y)**2

def gradSSE(Phi, Y, weights):
	return np.dot(np.transpose(Phi), np.dot(Phi, weights) - Y)

def ridge(Phi, Y, lamb):
	matInv = np.linalg.inv(np.dot(np.transpose(Phi), Phi) + lamb*np.eye(Phi.shape[1]))
	return np.dot(matInv, np.dot(np.transpose(Phi), Y))

ridge_weights = []
ridge_valid_loss = []
for lamb in lambs:
	weight = ridge(phi_train, Y_train, lamb)
	ridge_weights.append(weight)
	ridge_valid_loss.append(SSE(phi_val, Y_val, weight))
ridge_min = np.where(ridge_valid_loss == min(ridge_valid_loss))[0][0]
ridge_lamb = lambs[ridge_min]
ridge_weight = ridge_weights[ridge_min]
ridge_loss = ridge_valid_loss[ridge_min]

# Fit LASSO
def softThresh(x, lamb):
	M = len(x)
	thresh = []
	for d in range(M):
		if x[d] > lamb:
			thresh.append(x[d] - lamb)
		elif x[d] < -lamb:
			thresh.append(x[d] + lamb)
		else:
			thresh.append(0)
	return thresh

def lasso(Phi, Y, init, lamb, eps, max_iter):
	w = init
	max_diff = eps + 1
	t = 0
	while max_diff > eps and t < max_iter:
		new_w = softThresh(w - lamb * gradSSE(Phi, Y, w), lamb)
		max_diff = max(max_diff, max([abs(diff) for diff in (np.array(new_w) - np.array(w))]))
		w = new_w
		t += 1
	return w

lasso_weights = []
lasso_valid_loss = []
for lamb in lambs:
	weight = lasso(phi_train, Y_train, init, lamb, eps, max_iter)
	lasso_weights.append(weight)
	lasso_valid_loss.append(SSE(phi_val, Y_val, weight))
lasso_min = np.where(lasso_valid_loss == min(lasso_valid_loss))[0][0]
lasso_lamb = lambs[lasso_min]
lasso_weight = lasso_weights[lasso_min]
lasso_loss = lasso_valid_loss[lasso_min]

# Plots
def sinFunc(x, M, weight):
	phi = [x]
	for i in range(1, M + 1):
		phi.append(np.sin(0.4*np.pi*x*i))	
	return np.dot(phi, weight)

x_points = [float(x)/1000 for x in range(-1000,1000)]
y_points = [sinFunc(x, M, true_weight) for x in x_points]
ridge_points = [sinFunc(x, M, ridge_weight) for x in x_points]
lasso_points = [sinFunc(x, M, lasso_weight) for x in x_points]
MLE_points = [sinFunc(x, M, MLE_weight) for x in x_points]
fig = plt.figure()
plt.plot(X_test, Y_test, 'o')
#plt.plot(X_train, Y_train, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_points, y_points, label = "True")
plt.plot(x_points, ridge_points, label = "Ridge")
plt.plot(x_points, lasso_points, label = "LASSO")
plt.plot(x_points, MLE_points, label = "MLE")
plt.legend()
plt.savefig('hw1_4-1.pdf')
plt.show()

# Plot weights
w_points = [w+0.5 for w in range(13)]
x_values = [w+1 for w in range(13)]
fig = plt.figure()
plt.xticks(x_values)
plt.bar(w_points, true_weight, width = 1)
plt.savefig('hw1_4-2_1.pdf')

fig = plt.figure()
plt.xticks(x_values)
plt.bar(w_points, MLE_weight, width = 1)
plt.savefig('hw1_4-2_2.pdf')

fig = plt.figure()
plt.xticks(x_values)
plt.bar(w_points, ridge_weight, width = 1)
plt.savefig('hw1_4-2_3.pdf')

fig = plt.figure()
plt.xticks(x_values)
plt.bar(w_points, lasso_weight, width = 1)
plt.savefig('hw1_4-2_4.pdf')

# Plot test error
heights = [SSE(phi_test, Y_test, true_weight), SSE(phi_test, Y_test, MLE_weight), SSE(phi_test, Y_test, ridge_weight), SSE(phi_test, Y_test, lasso_weight)]
x_points = [x+0.7 for x in range(4)]
x_values = [1,2,3,4]
x_labels = ["True", "MLE", "Ridge", "LASSO"]
fig, ax = plt.subplots()
ax.bar(x_points, heights, width = 0.6)
ax.set_xticks(x_values)
ax.set_xticklabels(x_labels)
ax.set_ylabel('SSE')
ax.set_ylim([0, 1.6])
plt.savefig('hw1_4-1_2.pdf')

# Plot SSE against true weights
heights = [L2(MLE_weight, true_weight), L2(ridge_weight, true_weight), L2(lasso_weight, true_weight)]
x_points = [x+0.7 for x in range(3)]
x_values = [1,2,3]
x_labels = ["MLE", "Ridge", "LASSO"]
fig, ax = plt.subplots()
ax.bar(x_points, heights, width = 0.6)
ax.set_xticks(x_values)
ax.set_xticklabels(x_labels)
ax.set_ylabel('L2 distance')
plt.savefig('hw1_4-1_3.pdf')

