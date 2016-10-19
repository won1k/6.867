import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from svm import trainSVM, trainKernelSVM, predictSVM, predictKernelSVM
from pegasos import pegasos, kernelPegasos
import time

# Constants
train_sizes = [200, 300, 400, 500]
valid_size = 150
test_size = 150
norm = False

def gausRBF(x1, x2, gamma = 1):
	# assume x are vectors of size p
	return np.exp(-gamma*np.linalg.norm(x1-x2)**2)

# Load data functions
def data1v7(train_size):
	file1 = np.loadtxt("data/mnist_digit_1.csv")
	file2 = np.loadtxt("data/mnist_digit_7.csv")
	start = 0
	end = train_size
	X_train = np.concatenate((file1[start:end,:], file2[start:end,:]))
	y_train = np.array([1]*train_size + [-1]*train_size)
	start = end
	end += valid_size
	X_valid = np.concatenate((file1[start:end,:], file2[start:end,:]))
	y_valid = np.array([1]*valid_size + [-1]*valid_size)
	start = end
	end += test_size
	X_test = np.concatenate((file1[start:end,:], file2[start:end,:]))
	y_test = np.array([1]*test_size + [-1]*test_size)
	return X_train, y_train, X_valid, y_valid, X_test, y_test

def data3v5(train_size):
	file1 = np.loadtxt("data/mnist_digit_3.csv")
	file2 = np.loadtxt("data/mnist_digit_5.csv")
	start = 0
	end = train_size
	X_train = np.concatenate((file1[start:end,:], file2[start:end,:]))
	y_train = np.array([1]*train_size + [-1]*train_size)
	start = end
	end += valid_size
	X_valid = np.concatenate((file1[start:end,:], file2[start:end,:]))
	y_valid = np.array([1]*valid_size + [-1]*valid_size)
	start = end
	end += test_size
	X_test = np.concatenate((file1[start:end,:], file2[start:end,:]))
	y_test = np.array([1]*test_size + [-1]*test_size)
	return X_train, y_train, X_valid, y_valid, X_test, y_test

def data4v9(train_size):
	file1 = np.loadtxt("data/mnist_digit_4.csv")
	file2 = np.loadtxt("data/mnist_digit_9.csv")
	start = 0
	end = train_size
	X_train = np.concatenate((file1[start:end,:], file2[start:end,:]))
	y_train = np.array([1]*train_size + [-1]*train_size)
	start = end
	end += valid_size
	X_valid = np.concatenate((file1[start:end,:], file2[start:end,:]))
	y_valid = np.array([1]*valid_size + [-1]*valid_size)
	start = end
	end += test_size
	X_test = np.concatenate((file1[start:end,:], file2[start:end,:]))
	y_test = np.array([1]*test_size + [-1]*test_size)
	return X_train, y_train, X_valid, y_valid, X_test, y_test

def dataEvenvOdd(train_size):
	file1 = np.loadtxt("data/mnist_digit_0.csv")
	file2 = np.loadtxt("data/mnist_digit_7.csv")
	start = 0
	end = train_size/5
	X_train = file1[start:end,:]
	X_valid = file1[end:end+valid_size/5,:]
	X_test = file1[end+valid_size/5:end+valid_size/5+test_size/5,:]
	for even in ['2','4','6','8']:
		file1 = np.loadtxt("data/mnist_digit_" + even + ".csv")
		X_train = np.concatenate((X_train, file1[start:end,:]))
		X_valid = np.concatenate((X_valid, file1[end:end+valid_size/5,:]))
		X_test = np.concatenate((X_test, file1[end+valid_size/5:end+valid_size/5+test_size/5,:]))
	for odd in ['1','3','5','7','9']:
		file2 = np.loadtxt("data/mnist_digit_" + odd + ".csv")
		X_train = np.concatenate((X_train, file2[start:end,:]))
		X_valid = np.concatenate((X_valid, file2[end:end+valid_size/5,:]))
		X_test = np.concatenate((X_test, file2[end+valid_size/5:end+valid_size/5+test_size/5,:]))
	y_train = np.array([1]*train_size + [-1]*train_size)
	y_valid = np.array([1]*valid_size + [-1]*valid_size)
	y_test = np.array([1]*test_size + [-1]*test_size)
	return X_train, y_train, X_valid, y_valid, X_test, y_test

def normalizedData(X):
	return 2*X / 255 - 1


# Training logistic vs SVM


Cs = [10**i for i in range(-2,3)]
gamma = 1

test_acc_lin_QP = []
test_acc_lin_peg = []
test_acc_gaus_QP = []
test_acc_gaus_peg = []

time_lin_QP = []
time_lin_peg = []
time_gaus_QP = []
time_gaus_peg = []

for train_size in train_sizes:
	max_epoch = 20*train_size

	# Data
	X_train, y_train, X_valid, y_valid, X_test, y_test = data1v7(train_size)
	if norm:
		X_train = normalizedData(X_train)
		X_valid = normalizedData(X_valid)
		X_test = normalizedData(X_test)

	for C in Cs:
		lamb = 1.0/(train_size*C)

		# Train linear SVM (QP)
		start_time = time.time()
		QPlin = trainSVM(X_train, y_train, C)
		time_lin_QP.append(time.time() - start_time)

		# Train Gaussian SVM (QP)
		def kernel(x1, x2):
			return gausRBF(x1, x2, gamma)
		start_time = time.time()
		QPgaus = trainKernelSVM(kernel, X_train, y_train, C)
		time_gaus_QP.append(time.time() - start_time)

		# Train linear Pegasos
		start_time = time.time()
		peglin_w, peglin_w0 = pegasos(X_train, y_train, lamb, max_epoch)
		time_lin_peg.append(time.time() - start_time)

		# Train Gaussian Pegasos
		start_time = time.time()
		K_train = np.zeros((train_size*2, train_size*2))
		for i in range(train_size*2):
			for j in range(train_size*2):
				K_train[i][j] = kernel(X_train[i,:], X_train[j,:])
		peggaus = kernelPegasos(K_train, y_train, lamb, max_epoch)
		time_gaus_peg.append(time.time() - start_time)

		# Test accuracy
		acc_lin_QP = 0.0
		acc_gaus_QP = 0.0
		acc_lin_peg = 0.0
		acc_gaus_peg = 0.0

		def predict_linearSVM(x, w, w0):
			return(1 if np.dot(w, x) + w0 > 0 else -1)

		def predict_gaussianSVM(x, alpha, X, y):
			pred = 0
			for i in range(train_size*2):
				pred += alpha[i] * y[i] * gausRBF(X[i,:], x, gamma)
			return(1 if pred > 0 else -1)

		for i in range(test_size*2):
			if predictSVM(X_test[i,:], QPlin['x'], X_train, y_train.reshape(len(y_train),1)) == y_test[i]:
				acc_lin_QP += 1
			if predictKernelSVM(X_test[i,:], QPgaus['x'], kernel, X_train, y_train.reshape(len(y_train),1)) == y_test[i]:
				acc_gaus_QP += 1
			if predict_linearSVM(X_test[i,:], peglin_w, peglin_w0) == y_test[i]:
				acc_lin_peg += 1
			if predict_gaussianSVM(X_test[i,:], peggaus, X_train, y_train.reshape(len(y_train),1)) == y_test[i]:
				acc_gaus_peg += 1
		acc_lin_QP /= (2*test_size)
		acc_gaus_QP /= (2*test_size)
		acc_lin_peg /= (2*test_size)
		acc_gaus_peg /= (2*test_size)
		test_acc_lin_QP.append(acc_lin_QP)
		test_acc_gaus_QP.append(acc_gaus_QP)
		test_acc_lin_peg.append(acc_lin_peg)
		test_acc_gaus_peg.append(acc_gaus_peg)

# Plots
plt.figure()
plt.plot(train_size, [time_lin_QP[i] for i in range(0, len(20), 5)], label = "Linear, QP")
plt.plot(train_size, [time_lin_peg[i] for i in range(0, len(20), 5)], label = "Linear, Pegasos")
plt.plot(train_size, [time_gaus_QP[i] for i in range(0, len(20), 5)], label = "Gaus, QP")
plt.plot(train_size, [time_gaus_peg[i] for i in range(0, len(20), 5)], label = "Gaus, Pegasos")
plt.xlabel("Trainset Size")
plt.ylabel("Running Time (s)")
plt.savefig("hw2_4-3_time.pdf")

plotT = [[t]*len(Cs) for t in train_sizes]
plotT = [item for sublist in plotT for item in sublist]
plotC = np.log(Cs*len(train_sizes))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(plotT, plotC, test_acc_lin_QP)
ax.plot_trisurf(plotT, plotC, test_acc_lin_peg)
ax.set_xlabel('Trainset Size')
ax.set_ylabel('Log(C)')
ax.set_zlabel('Test Accuracy')
plt.savefig('hw2_4-3_acc_lin.pdf')

plotT = [[t]*len(Cs) for t in train_sizes]
plotT = [item for sublist in plotT for item in sublist]
plotC = np.log(Cs*len(train_sizes))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(plotT, plotC, test_acc_gaus_QP)
ax.plot_trisurf(plotT, plotC, test_acc_gaus_peg)
ax.set_xlabel('Trainset Size')
ax.set_ylabel('Log(C)')
ax.set_zlabel('Test Accuracy')
plt.savefig('hw2_4-3_acc_gaus.pdf')











