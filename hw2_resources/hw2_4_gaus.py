import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from svm import trainSVM, trainKernelSVM, predictSVM, predictKernelSVM
from pegasos import pegasos, kernelPegasos
from sklearn.linear_model import LogisticRegression

# Constants
train_size = 200
valid_size = 150
test_size = 150
norm = False

def gausRBF(x1, x2, gamma = 1):
	# assume x are vectors of size p
	return np.exp(-gamma*np.linalg.norm(x1-x2)**2)

# Load data functions
def data1v7():
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

def data3v5():
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

def data4v9():
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

def dataEvenvOdd():
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
X_train, y_train, X_valid, y_valid, X_test, y_test = data1v7()
if norm:
	X_train = normalizedData(X_train)
	X_valid = normalizedData(X_valid)
	X_test = normalizedData(X_test)

Cs = [10**i for i in range(-2,3)]
gammas = [10**i for i in range(-3,3)]

valid_acc_LR1 = 0
valid_acc_LR2 = 0
valid_acc_SVM = 0
valid_acc_gaus = 0
valid_best_LR1 = 0
valid_best_LR2 = 0
valid_best_SVM = 0
valid_best_gaus_C = 0
valid_best_gaus_gam = 0

for C in Cs:
	# Train 
	LR1 = LogisticRegression(penalty = 'l1', C = C).fit(X_train, y_train)
	LR2 = LogisticRegression(penalty = 'l2', C = C).fit(X_train, y_train)
	SVMsol = trainSVM(X_train, y_train, C)

	# Validation accuracy
	if LR1.score(X_valid, y_valid) > valid_acc_LR1:
		valid_acc_LR1 = LR1.score(X_valid, y_valid)
		valid_best_LR1 = C
	if LR2.score(X_valid, y_valid) > valid_acc_LR2:
		valid_acc_LR2 = LR2.score(X_valid, y_valid)
		valid_best_LR2 = C
	acc_SVM = 0.0
	for i in range(valid_size*2):
		if predictSVM(X_valid[i,:], SVMsol['x'], X_train, y_train.reshape(len(y_train),1)) == y_valid[i]:
			acc_SVM += 1
	if acc_SVM / (2*valid_size) > valid_acc_SVM:
		valid_acc_SVM = acc_SVM / (2*valid_size)
		valid_best_SVM = C

	for gamma in gammas:
		# Train classifier Gaussian
		def kernel(x1, x2):
			return gausRBF(x1, x2, gamma)
		GausSVM = trainKernelSVM(kernel, X_train, y_train, C)

		# Validation
		acc_gaus = 0.0
		for i in range(valid_size*2):
			if predictKernelSVM(X_valid[i,:], GausSVM['x'], kernel, X_train, y_train.reshape(len(y_train),1)) == y_valid[i]:
				acc_gaus += 1
		if acc_gaus / (2*valid_size) > valid_acc_gaus:
			valid_acc_gaus = acc_gaus / (2*valid_size)
			valid_best_gaus_C = C
			valid_best_gaus_gam = gamma
		
# Test acc for best model
LR1 = LogisticRegression(penalty = 'l1', C = valid_best_LR1).fit(X_train, y_train)
LR2 = LogisticRegression(penalty = 'l2', C = valid_best_LR2).fit(X_train, y_train)
SVMsol = trainSVM(X_train, y_train, valid_best_SVM)
def kernel(x1, x2):
	return gausRBF(x1, x2, valid_best_gaus_gam)
GausSVM = trainKernelSVM(kernel, X_train, y_train, valid_best_gaus_C)

test_acc_LR1 = LR1.score(X_test, y_test)
test_acc_LR2 = LR2.score(X_test, y_test)
test_acc_SVM = 0.0
for i in range(test_size*2):
	if predictSVM(X_test[i,:], SVMsol['x'], X_train, y_train.reshape(len(y_train),1)) == y_test[i]:
		test_acc_SVM += 1
test_acc_SVM /= (2*test_size)
test_acc_gaus = 0.0
for i in range(test_size*2):
	if predictKernelSVM(X_test[i,:], GausSVM['x'], kernel, X_train, y_train.reshape(len(y_train),1)) == y_test[i]:
		test_acc_gaus += 1
test_acc_gaus /= (2*test_size)

# Plot
w_points = [w+0.75 for w in range(4)]
x_values = [w+1 for w in range(4)]
fig = plt.figure()
plt.xticks(x_values)
plt.bar(w_points, [test_acc_LR1, test_acc_LR2, test_acc_SVM, test_acc_gaus], width = 0.5)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.ylim([0.85, 1])
plt.savefig('hw2_4-2_test.pdf')

fig = plt.figure()
plt.xticks(x_values)
plt.bar(w_points, [valid_acc_LR1, valid_acc_LR2, valid_acc_SVM, valid_acc_gaus], width = 0.5)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.ylim([0.85, 1])
plt.savefig('hw2_4-2_valid.pdf')

print(valid_best_gaus_gam)












