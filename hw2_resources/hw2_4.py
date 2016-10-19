import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from svm import trainSVM, predictSVM
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
train_acc_L1 = []
train_acc_L2 = []
train_acc_SVM = []
test_acc_L1 = []
test_acc_L2 = []
test_acc_SVM = []
test_err_L1 = []
test_err_L2 = []
test_err_SVM = []
for C in Cs:
	#lamb = 1/(train_size*C)
	# Train classifiers
	LR1 = LogisticRegression(penalty = 'l1', C = C).fit(X_train, y_train)
	LR2 = LogisticRegression(penalty = 'l2', C = C).fit(X_train, y_train)
	SVMsol = trainSVM(X_train, y_train, C)
	# Train acc
	train_acc_L1.append(LR1.score(X_train, y_train))
	train_acc_L2.append(LR2.score(X_train, y_train))
	acc = 0.0
	for i in range(train_size*2):
		if predictSVM(X_train[i,:], SVMsol['x'], X_train, y_train.reshape(len(y_train),1)) == y_train[i]:
			acc += 1.0
	train_acc_SVM.append(acc / (2*train_size))
	# Test acc
	#test_acc_L1.append(LR1.score(X_test, y_test))
	#test_acc_L2.append(LR2.score(X_test, y_test))
	acc_L1 = 0.0
	acc_L2 = 0.0
	acc_SVM = 0.0
	for i in range(test_size*2):
		if LR1.predict(X_test[i,:])[0] == y_test[i]:
			acc_L1 += 1.0
		else:
			test_err_L1.append(i)
		if LR2.predict(X_test[i,:])[0] == y_test[i]:
			acc_L2 += 1.0
		else:
			test_err_L2.append(i)
		if predictSVM(X_test[i,:], SVMsol['x'], X_train, y_train.reshape(len(y_train),1)) == y_test[i]:
			acc_SVM += 1.0
		else:
			test_err_SVM.append(i)
	test_acc_L1.append(acc_L1 / (2*test_size))
	test_acc_L2.append(acc_L2 / (2*test_size))
	test_acc_SVM.append(acc_SVM / (2*test_size))

# Plotting
fig = plt.figure()
rand = np.random.randn(1)/1e4
plt.plot(np.log(Cs), [acc + rand for acc in train_acc_L1], label = "L1, Train")
rand = np.random.randn(1)/1e4
plt.plot(np.log(Cs), [acc + rand for acc in test_acc_L1], label = "L1, Test")
rand = np.random.randn(1)/1e4
plt.plot(np.log(Cs), [acc + rand for acc in train_acc_L2], label = "L2, Train")
rand = np.random.randn(1)/1e4
plt.plot(np.log(Cs), [acc + rand for acc in test_acc_L2], label = "L2, Test")
rand = np.random.randn(1)/1e4
plt.plot(np.log(Cs), [acc + rand for acc in train_acc_SVM], label = "SVM, Train")
rand = np.random.randn(1)/1e4
plt.plot(np.log(Cs), [acc + rand for acc in test_acc_SVM], label = "SVM, Test")
plt.xlabel('Log(C)')
plt.ylabel('Accuracy')
#plt.ylim([0.98, 1.002])
plt.legend()
#plt.savefig('hw2_4-1_1v7_norm.pdf')

# Errors
print(set(test_err_L1))
print(set(test_err_L2))
print(set(test_err_SVM))

# 1v7: 62
# 3v5:
# 4v9:
# evo:










