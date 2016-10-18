import numpy as np
from cvxopt import matrix, solvers

def trainSVM(X, y, C):
	n = len(y)
	try:
		flaty = [v[0] for v in y]
	except:
		flaty = y
	Y = np.diag(flaty)
	K = np.dot(X, np.transpose(X))

	# define your matrices
	P = matrix(np.dot(Y, np.dot(K, Y)), (n,n), tc = 'd')
	q = matrix(-np.ones(n), tc='d')
	G = matrix(np.concatenate((np.eye(n), -np.eye(n))), tc='d')
	h = matrix(np.concatenate((np.transpose(np.matrix([C]*n)), np.transpose(np.matrix([0]*n)))), tc = 'd')
	A = matrix([[v] for v in flaty], tc='d')
	b = matrix(0.0, tc='d')
	
	# find the solution	
	solution = solvers.qp(P, q, G, h, A, b)
	print(solution['status'])
	#xvals = np.array(solution['x'])
	return solution

def trainKernelSVM(K, X, y, C):
	n = len(y)
	try:
		flaty = [v[0] for v in y]
	except:
		flaty = y
	Y = np.diag(flaty)

	# if X is None, K is matrix; else K is function
	if X is not None:
		KMatrix = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				KMatrix[i][j] = flaty[i]*flaty[j]*K(X[i,:],X[j,:])
		P = matrix(np.dot(Y, np.dot(KMatrix, Y)), (n,n), tc = 'd')
	else:
		P = matrix(np.dot(Y, np.dot(K, Y)), (n,n), tc = 'd')

	# Constraints
	q = matrix(-np.ones(n), tc='d')
	G = matrix(np.concatenate((np.eye(n), -np.eye(n))), tc='d')
	h = matrix(np.concatenate((np.transpose(np.matrix([C]*n)), np.transpose(np.matrix([0]*n)))), tc = 'd')
	A = matrix([[v] for v in flaty], tc='d')
	b = matrix(0.0, tc='d')
	return solvers.qp(P, q, G, h, A, b)

def predictSVM(x, a, X, y):
	n = len(y)
	w = np.sum(a*y*X,0)
	sv = [i for i in range(n) if a[i] > 1e-5]
	b = 0
	K = np.dot(X, np.transpose(X))
	for i in sv:
		med = np.dot(np.transpose(a*y),K[i,:])
		b += y[i] - med
	b /= len(sv)

	pred = np.dot(w,x) + b
	return(1 if pred > 0 else -1)