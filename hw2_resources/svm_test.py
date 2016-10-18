from numpy import *
from plotBoundary import *
import pylab as pl
import matplotlib.pyplot as plt
from svm import trainSVM, predictSVM

def predictAccuracy(X, y, scoreFn):
	n = len(y)
	accuracy = 0.0
	for i, t in enumerate(y):
		if scoreFn(X[i,:]) == t:
			accuracy += 1.0
	return accuracy / n


acc = []
for name in ['1','2','3','4']:
	print(name)
	# parameters
	print '======Training======'
	# load data from csv files
	train = loadtxt('data/data'+name+'_train.csv')
	# use deep copy here to make cvxopt happy
	X_train = train[:, 0:2].copy()
	y_train = train[:, 2:3].copy()

	# Carry out training, primal and/or dual
	sol = trainSVM(X_train, y_train, 1)
	alpha = sol['x']

	def predict(x):
		return predictSVM(x, alpha, X_train, y_train)

	# plot training results
	plotDecisionBoundary(X_train, y_train, predict, [-1, 0, 1], title = 'SVM Train, Dataset ' + name)
	pl.savefig('hw2_2-2_' + name + '_train.pdf')


	print '======Validation======'
	# load data from csv files
	validate = loadtxt('data/data'+name+'_validate.csv')
	X = validate[:, 0:2]
	y = validate[:, 2:3]
	# plot validation results
	plotDecisionBoundary(X, y, predict, [-1, 0, 1], title = 'SVM Validate, Dataset ' + name)
	pl.savefig('hw2_2-2_' + name + '_validate.pdf')

	print '======Test======'
	# load data from csv files
	validate = loadtxt('data/data'+name+'_test.csv')
	X = validate[:, 0:2]
	y = validate[:, 2:3]
	# plot validation results
	plotDecisionBoundary(X, y, predict, [-1, 0, 1], title = 'SVM Test, Dataset ' + name)
	pl.savefig('hw2_2-2_' + name + '_test.pdf')
	acc.append(predictAccuracy(X, y, predict))

# Plot accuracy
print(acc)
w_points = [w+0.75 for w in range(4)]
x_values = [w+1 for w in range(4)]
fig = plt.figure()
plt.xticks(x_values)
plt.bar(w_points, acc, width = 0.5)
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.savefig('hw2_2-2_acc.pdf')
