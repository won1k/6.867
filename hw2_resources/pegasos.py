import numpy as np

def pegasos(X, y, lamb, epochs):
	# Setup
	n = len(y)
	t = 0
	w = np.zeros(X.shape[1])
	w0 = 0

	# Training
	for epoch in range(epochs):
		for i in range(n):
			t += 1
			eta = 1/(t*lamb)
			if y[i] * (np.dot(w, X[i,:])+w0) < 1:
				w = (1-eta*lamb)*w + eta*y[i]*X[i,:]
				w0 = w0 + eta*y[i]
			else:
				w = (1-eta*lamb)*w

	return w, w0

def kernelPegasos(K, y, lamb, epochs):
	# Setup
	n = len(y)
	t = 0
	alpha = np.zeros(n)

	# Training
	for epoch in range(epochs):
		for i in range(n):
			t += 1
			eta = 1/(t*lamb)
			if y[i] * np.sum(np.dot(K[i,:],alpha)) < 1:
				alpha[i] = (1-eta*lamb)*alpha[i] + eta*y[i]
			else:
				alpha[i] = (1-eta*lamb)*alpha[i]
	
	return alpha